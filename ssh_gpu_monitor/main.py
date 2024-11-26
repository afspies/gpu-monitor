"""This module is used to check GPU machines information asynchronously."""

import asyncio
import os
from typing import NamedTuple, Dict, List, Any
import xml.etree.ElementTree as ET
import re
import sys
import logging
from logging.handlers import RotatingFileHandler
from rich.live import Live
from rich.console import Console
import signal
from pathlib import Path

# Add these imports at the top
import asyncssh
from .src.table_display import GPUTable
from .src.config_loader import generate_targets, Target

class GPUInfo(NamedTuple):
    """A class for representing a GPU machine's information."""

    model: str
    num_procs: int
    gpu_util: str
    used_mem: str
    total_mem: str

    def __str__(self) -> str:
        return (
            f'{self.model:26} | '
            f'Free: {self.num_procs == 0!s:5} | '
            f'Num Procs: {self.num_procs:2d} | '
            f'GPU Util: {self.gpu_util:>3} % | '
            f'Memory: {self.used_mem:>5} / {self.total_mem:>5} MiB'
        )

class AsyncGPUChecker:
    """A class for asynchronously checking GPU machines."""

    def __init__(self, targets: List[Target]) -> None:
        self.targets = targets
        self.proc_filter = re.compile(r'.*')
        self.gpu_table = GPUTable()
        self.connections = {}
        self.jump_conn = None
        self.console = Console()
        self.running = True
        # Add semaphore for connection pooling
        self.connection_semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent connections

    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\nShutting down gracefully...")
        self.gpu_table.show_goodbye()  # Show goodbye message
        self.running = False
        
    async def open_connection(self, target: Target):
        """Modified connection method with better error handling and rate limiting"""
        self.gpu_table.add_status(f"[{target.host}] Waiting for connection semaphore", "cyan")
        logging.debug(f"[{target.host}] Waiting for connection semaphore (current limit: 10)")
        async with self.connection_semaphore:  # Use semaphore to limit concurrent connections
            try:
                self.gpu_table.add_status(f"[{target.host}] Attempting port forwarding", "yellow")
                logging.debug(f"[{target.host}] Attempting port forwarding through jump host")
                # Set up port forwarding
                try:
                    listener = await asyncio.wait_for(
                        self.jump_conn.forward_local_port('', 0, target.host, 22),
                        timeout=SSH_TIMEOUT/2
                    )
                    tunnel_port = listener.get_port()
                    self.gpu_table.add_status(f"[{target.host}] Port forwarding established", "green")
                    logging.debug(f"[{target.host}] Port forwarding established on port {tunnel_port}")
                except asyncio.TimeoutError:
                    error_msg = f"Port forwarding timeout after {SSH_TIMEOUT/2} seconds"
                    self.gpu_table.add_status(f"[{target.host}] {error_msg}", "red")
                    logging.error(f"[{target.host}] {error_msg}")
                    return {target.host: error_msg}
                except Exception as e:
                    error_msg = f"Port forwarding error: {str(e)}"
                    self.gpu_table.add_status(f"[{target.host}] {error_msg}", "red")
                    logging.error(f"[{target.host}] {error_msg}", exc_info=True)
                    return {target.host: error_msg}
                
                # Expand the key path
                key_path = os.path.expanduser(target.key_path)
                self.gpu_table.add_status(f"[{target.host}] Attempting SSH connection", "yellow")
                logging.debug(f"[{target.host}] Using SSH key: {key_path}")
                
                if not os.path.exists(key_path):
                    error_msg = f"SSH key not found: {key_path}"
                    self.gpu_table.add_status(f"[{target.host}] {error_msg}", "red")
                    logging.error(f"[{target.host}] {error_msg}")
                    return {target.host: error_msg}
                
                # Attempt SSH connection using target-specific username and key
                try:
                    conn = await asyncio.wait_for(
                        asyncssh.connect(
                            'localhost', 
                            port=tunnel_port,
                            username=target.username,
                            client_keys=[key_path],  # Use target-specific key
                            known_hosts=None,
                            keepalive_interval=30,
                            keepalive_count_max=5
                        ),
                        timeout=SSH_TIMEOUT/2
                    )
                    
                    self.connections[target.host] = conn
                    self.gpu_table.add_status(f"[{target.host}] Connection established", "green")
                    logging.debug(f"[{target.host}] Successfully established SSH connection")
                    return {target.host: "Connected"}
                    
                except asyncio.TimeoutError:
                    error_msg = f"SSH connection timeout after {SSH_TIMEOUT/2} seconds"
                    self.gpu_table.add_status(f"[{target.host}] {error_msg}", "red")
                    logging.error(f"[{target.host}] {error_msg}")
                    return {target.host: error_msg}
                except asyncssh.Error as e:
                    error_msg = f"SSH Error: {str(e)}"
                    self.gpu_table.add_status(f"[{target.host}] {error_msg}", "red")
                    logging.error(f"[{target.host}] {error_msg}", exc_info=True)
                    return {target.host: error_msg}
                except Exception as e:
                    error_msg = f"Unexpected connection error: {str(e)}"
                    self.gpu_table.add_status(f"[{target.host}] {error_msg}", "red")
                    logging.error(f"[{target.host}] {error_msg}", exc_info=True)
                    return {target.host: error_msg}
                    
            except Exception as e:
                error_msg = f"Critical connection error: {str(e)}"
                self.gpu_table.add_status(f"[{target.host}] {error_msg}", "red")
                logging.error(f"[{target.host}] {error_msg}", exc_info=True)
                return {target.host: error_msg}

    async def check_single_target(self, target: Target) -> Dict[str, str]:
        """Check a single GPU target using an existing connection."""
        logging.debug(f"[{target.host}] Starting GPU status check")
        try:
            conn = self.connections.get(target.host)
            if not conn:
                logging.warning(f"[{target.host}] No active connection found")
                return {target.host: "No connection"}
            
            logging.debug(f"[{target.host}] Running nvidia-smi command")
            result = await asyncio.wait_for(
                conn.run('nvidia-smi -q -x', check=True),
                timeout=SSH_TIMEOUT
            )
            if result.exit_status != 0:
                logging.error(f"[{target.host}] Command failed with status {result.exit_status}: {result.stderr}")
                return {target.host: f"Command failed: {result.stderr}"}
            
            # Ensure we're working with a string
            output = result.stdout
            if isinstance(output, bytes):
                output = output.decode('utf-8')
            
            logging.debug(f"[{target.host}] Successfully received nvidia-smi output, length: {len(output)} bytes")
            return {target.host: self.parse_gpu_info(target.host, output)}
        except asyncio.TimeoutError:
            logging.error(f"[{target.host}] Query timeout after {SSH_TIMEOUT} seconds")
            return {target.host: "Timeout"}
        except asyncssh.Error as exc:
            logging.error(f"[{target.host}] SSH Error: {str(exc)}", exc_info=True)
            return {target.host: f"SSH Error: {str(exc)}"}
        except Exception as exc:
            logging.error(f"[{target.host}] Unexpected error: {str(exc)}", exc_info=True)
            return {target.host: f"Unexpected error: {str(exc)}"}

    def parse_gpu_info(self, machine_name: str, xml_output: str) -> str:
        """Parse the GPU info from XML output."""
        logging.debug(f"[{machine_name}] Parsing GPU information from XML")
        try:
            # Ensure xml_output is a string
            if isinstance(xml_output, bytes):
                xml_output = xml_output.decode('utf-8')
            root = ET.fromstring(xml_output)

            gpu_infos = []
            for i, gpu in enumerate(root.findall('gpu')):
                try:
                    logging.debug(f"[{machine_name}] Parsing GPU {i}")
                    model = gpu.find('product_name').text
                    processes = gpu.find('processes')
                    
                    # More robust process counting
                    num_procs = 0
                    if processes is not None:
                        for process in processes.findall('process_info'):
                            proc_name = process.find('process_name')
                            if proc_name is not None and proc_name.text is not None:
                                if self.proc_filter.search(proc_name.text):
                                    num_procs += 1
                    
                    gpu_util = gpu.find('utilization').find('gpu_util').text.removesuffix(' %')
                    memory_usage = gpu.find('fb_memory_usage')
                    used_mem = memory_usage.find('used').text.removesuffix(' MiB')
                    total_mem = memory_usage.find('total').text.removesuffix(' MiB')
                    
                    gpu_info = GPUInfo(model, num_procs, gpu_util, used_mem, total_mem)
                    logging.debug(f"[{machine_name}] GPU {i} info: {gpu_info}")
                    gpu_infos.append(gpu_info)
                except AttributeError as e:
                    logging.error(f"[{machine_name}] Error parsing GPU {i} info: {str(e)}", exc_info=True)
                    return f"Error parsing GPU info: {str(e)}"

            # Join multiple GPU infos with newlines
            return '\n'.join(map(str, gpu_infos))
        except ET.ParseError as e:
            logging.error(f"[{machine_name}] XML parse error: {str(e)}", exc_info=True)
            return f"XML parse error: {str(e)}"
        except Exception as e:
            logging.error(f"[{machine_name}] Unexpected error parsing GPU info: {str(e)}", exc_info=True)
            return f"Parse error: {str(e)}"

    async def run(self) -> None:
        """Run the main loop of the GPU checker asynchronously."""
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print('\n-------------------------------------------------\n')
        self.gpu_table.add_status("Starting GPU checker...", "green")
        logging.info("Starting GPU checker")
        
        try:
            # Initialize table with "Connecting" status
            initial_data = {target.host: "Connecting" for target in self.targets}
            self.gpu_table.update_table(initial_data)
            self.gpu_table.add_status(f"Initialized with {len(self.targets)} targets", "blue")
            logging.info(f"Initialized table with {len(self.targets)} targets")

            # Create live display
            with Live(self.gpu_table.layout, console=self.console, refresh_per_second=4) as live:
                try:
                    # Open jump host connection
                    self.gpu_table.add_status(f"Connecting to jump host {JUMP_SHELL}", "yellow")
                    logging.info(f"Connecting to jump host {JUMP_SHELL} as {USERNAME}")
                    self.jump_conn = await asyncio.wait_for(
                        asyncssh.connect(
                            JUMP_SHELL, 
                            username=USERNAME, 
                            client_keys=[SSH_KEY_PATH],
                            keepalive_interval=30,
                            keepalive_count_max=5
                        ),
                        timeout=SSH_TIMEOUT
                    )
                    self.gpu_table.add_status("Successfully connected to jump host", "green")
                    logging.info("Successfully connected to jump host")

                    # Open connections to GPU servers in batches
                    total_targets = len(self.targets)
                    self.gpu_table.add_status(f"Opening connections to {total_targets} GPU servers", "blue")
                    logging.info(f"Starting to open connections to {total_targets} GPU servers")
                    
                    batch_size = 10
                    successful_connections = 0
                    failed_connections = 0
                    
                    for i in range(0, total_targets, batch_size):
                        batch = self.targets[i:i+batch_size]
                        batch_num = i//batch_size + 1
                        self.gpu_table.add_status(f"Processing batch {batch_num} ({len(batch)} targets)", "yellow")
                        logging.info(f"Processing batch {batch_num} ({len(batch)} targets)")
                        
                        connection_tasks = [self.open_connection(target) for target in batch]
                        connection_results = await asyncio.gather(*connection_tasks, return_exceptions=True)
                        
                        # Process connection results
                        batch_successes = 0
                        for target, result in zip(batch, connection_results):
                            if isinstance(result, dict):
                                self.gpu_table.update_table(result)
                                batch_successes += 1
                                successful_connections += 1
                            else:
                                failed_connections += 1
                                error_msg = str(result) if result is not None else "Unknown error"
                                self.gpu_table.update_table({target.host: f"Error: {error_msg}"})
                        
                        progress = (i + len(batch)) / total_targets * 100
                        self.gpu_table.add_status(
                            f"Progress: {progress:.1f}% ({successful_connections} ok, {failed_connections} failed)", 
                            "blue"
                        )
                        logging.info(f"Batch {batch_num} complete: {batch_successes}/{len(batch)} successful connections")
                        live.update(self.gpu_table.layout)
                        await asyncio.sleep(1)  # Brief pause between batches
                    
                    self.gpu_table.add_status(
                        f"Connection phase complete: {successful_connections} ok, {failed_connections} failed",
                        "green" if failed_connections == 0 else "yellow"
                    )
                    logging.info(f"Connection phase complete - Success: {successful_connections}, Failed: {failed_connections}")
                    
                except asyncio.TimeoutError:
                    error_msg = f"Timeout connecting to jump host after {SSH_TIMEOUT} seconds"
                    self.gpu_table.add_status(error_msg, "red")
                    logging.error(error_msg)
                    return
                    
                except Exception as e:
                    error_msg = f"Error connecting to jump host: {str(e)}"
                    self.gpu_table.add_status(error_msg, "red")
                    logging.error(error_msg, exc_info=True)
                    return

                # Start the query loop
                while self.running:
                    try:
                        query_tasks = [self.check_single_target(target) for target in self.targets]
                        query_results = await asyncio.gather(*query_tasks, return_exceptions=True)
                        
                        successful_queries = 0
                        failed_queries = 0
                        
                        for target, result in zip(self.targets, query_results):
                            if isinstance(result, dict):
                                self.gpu_table.update_table(result)
                                successful_queries += 1
                            else:
                                failed_queries += 1
                                error_msg = str(result) if result is not None else "Unknown error"
                                self.gpu_table.update_table({target.host: f"Query Error: {error_msg}"})
                        
                        self.gpu_table.add_status(
                            f"Query complete: {successful_queries} ok, {failed_queries} failed",
                            "green" if failed_queries == 0 else "yellow"
                        )
                        logging.info(f"Query cycle complete - Success: {successful_queries}, Failed: {failed_queries}")
                        
                        live.update(self.gpu_table.layout)
                        await asyncio.sleep(REFRESH_RATE)
                        
                    except Exception as e:
                        error_msg = f"Error in query loop: {str(e)}"
                        self.gpu_table.add_status(error_msg, "red")
                        logging.error(error_msg, exc_info=True)
                        await asyncio.sleep(REFRESH_RATE)  # Wait before retrying
                
        except Exception as e:
            error_msg = f"Critical error in main loop: {str(e)}"
            self.gpu_table.add_status(error_msg, "red")
            logging.error(error_msg, exc_info=True)
            
        finally:
            # Cleanup
            logging.info("Closing connections")
            for conn in self.connections.values():
                try:
                    conn.close()
                except:
                    pass
            if self.jump_conn:
                try:
                    self.jump_conn.close()
                except:
                    pass
            logging.info("GPU checker stopped")

    def _process_query_results(self, results):
        """Helper method to process query results and update the table"""
        data = {}
        for result in results:
            if isinstance(result, dict):
                data.update(result)
            else:
                logging.error(f"Query error: {str(result)}")
        self.gpu_table.update_table(data)

def setup_logging(config: Dict) -> None:
    """Set up logging configuration based on debug config"""
    # Suppress all loggers initially
    logging.getLogger().setLevel(logging.CRITICAL)
    asyncssh_logger = logging.getLogger('asyncssh')
    asyncssh_logger.setLevel(logging.CRITICAL)

    if not config['debug']['enabled']:
        return

    try:
        # Log directory and file paths are now absolute from config_loader
        log_file = config['debug']['log_file']
        log_dir = os.path.dirname(log_file)
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a more detailed log format
        log_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # Set up file handler with rotation and immediate flush
        log_handler = RotatingFileHandler(
            log_file, 
            maxBytes=config['debug']['log_max_size'], 
            backupCount=config['debug']['log_backup_count'],
            mode='a'  # Append mode
        )
        log_handler.setFormatter(log_formatter)
        log_handler.setLevel(logging.DEBUG)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
        
        # Remove any existing handlers to prevent duplicate logging
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        root_logger.addHandler(log_handler)
        
        # Add console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(console_handler)
        
        # Log initial debug information
        logging.debug(f"Log file initialized at: {log_file}")
        logging.debug(f"Debug configuration: {config['debug']}")
        
        # Force flush
        log_handler.flush()
        
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        # Set up console logging as fallback
        logging.basicConfig(
            level=logging.DEBUG if config['debug']['enabled'] else logging.CRITICAL,
            format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )

async def main(config: Dict[str, Any]) -> None:
    """Main function for running GPU checker."""
    # Remove config loading since it's now passed in
    setup_logging(config)
    
    # Expand SSH key path
    config['ssh']['key_path'] = os.path.expanduser(config['ssh']['key_path'])
    
    if config['debug']['enabled']:
        logging.info("GPU checker started in debug mode")
    
    if not os.path.exists(config['ssh']['key_path']):
        if config['debug']['enabled']:
            logging.error(f'SSH key not found at {config["ssh"]["key_path"]}')
        print('SSH key not found. Please check the provided path.')
        return

    # Generate target list
    targets = generate_targets(config)
    
    if config['debug']['enabled']:
        logging.info(f"Generated targets: {targets}")

    # Update global constants
    global USERNAME, SSH_KEY_PATH, JUMP_SHELL, SSH_TIMEOUT, REFRESH_RATE
    USERNAME = config['ssh']['username']
    SSH_KEY_PATH = config['ssh']['key_path']
    JUMP_SHELL = config['ssh']['jump_host']
    SSH_TIMEOUT = config['ssh']['timeout']
    REFRESH_RATE = config['display']['refresh_rate']

    gpu_checker = AsyncGPUChecker(targets)
    await gpu_checker.run()

if __name__ == '__main__':
    asyncio.run(main())

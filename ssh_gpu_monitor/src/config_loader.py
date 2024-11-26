import yaml
import argparse
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

def get_default_config_path() -> Path:
    """Return the default config path for the package."""
    return Path(__file__).parent.parent / 'config' / 'config.yaml' 

def get_project_root() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).parent.parent.parent

class Target(NamedTuple):
    """Represents a target with its associated username and key path"""
    host: str
    username: str
    key_path: str

def generate_targets(config: Dict[str, Any]) -> List[Target]:
    """Generate list of targets from config specification."""
    targets = []
    default_username = config['ssh']['username']
    default_key_path = config['ssh']['key_path']
    
    # Add individual targets
    if 'individual' in config['targets']:
        for target in config['targets']['individual']:
            if isinstance(target, str):
                # Simple string target uses defaults
                targets.append(Target(target, default_username, default_key_path))
            else:
                # Dictionary target may have overrides
                targets.append(Target(
                    target['host'],
                    target.get('username', default_username),
                    target.get('key_path', default_key_path)
                ))
    
    # Add pattern-based targets
    if 'patterns' in config['targets']:
        for pattern in config['targets']['patterns']:
            pattern_username = pattern.get('username', default_username)
            pattern_key_path = pattern.get('key_path', default_key_path)
            targets.extend([
                Target(
                    pattern['format'].format(
                        prefix=pattern['prefix'],
                        number=x
                    ),
                    pattern_username,
                    pattern_key_path
                )
                for x in range(pattern['start'], pattern['end'] + 1)
            ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_targets = []
    for target in targets:
        if target.host not in seen:
            seen.add(target.host)
            unique_targets.append(target)
    
    return unique_targets

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SSH GPU Monitor')
    
    # Core arguments
    parser.add_argument('--get_config_path', action='store_true', 
                       help='Print the default config path and exit')
    parser.add_argument('--config', '-c', type=str,
                       help='Path to custom config file')
    
    # SSH configuration
    parser.add_argument('--ssh.username', type=str, help='Default SSH username')
    parser.add_argument('--ssh.key_path', type=str, help='SSH key path')
    parser.add_argument('--ssh.jump_host', type=str, help='Jump host')
    parser.add_argument('--ssh.timeout', type=int, help='SSH timeout in seconds')
    
    # Target configuration
    parser.add_argument('--targets', type=str, nargs='+', 
                       help='Override all targets with specified list (will use default username)')
    
    # Display configuration
    parser.add_argument('--display.refresh_rate', type=int, help='Refresh rate in seconds')
    
    # Debug configuration
    parser.add_argument('--debug.enabled', action='store_true', help='Enable debug mode')
    parser.add_argument('--debug.log_dir', type=str, help='Log directory')
    parser.add_argument('--debug.log_file', type=str, help='Log file name')
    
    return parser.parse_args()

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file and override with command line arguments."""
    args = parse_args()
    
    # Handle --get_config_path
    if args.get_config_path:
        print(get_default_config_path())
        import sys
        sys.exit(0)
    
    # Use provided config path, CLI config path, or default
    if config_path is None:
        config_path = args.config if args.config else get_default_config_path()
    
    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    for arg, value in vars(args).items():
        if value is not None and arg not in ['get_config_path', 'config']:
            if arg == 'targets':
                # Special handling for targets override
                config['targets'] = {
                    'individual': [{'host': t} for t in value],
                    'patterns': []
                }
            else:
                # Split the argument name into sections
                sections = arg.split('.')
                
                # Navigate through the config dictionary
                current = config
                for section in sections[:-1]:
                    current = current[section]
                
                # Set the value
                current[sections[-1]] = value
    
    # Make log paths absolute using project root
    if config['debug']['enabled']:
        project_root = get_project_root()
        # Ensure log_dir is an absolute path
        log_dir = Path(config['debug']['log_dir'])
        if not log_dir.is_absolute():
            log_dir = project_root / log_dir
        
        # Create log directory
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config with absolute paths
        config['debug']['log_dir'] = str(log_dir)
        config['debug']['log_file'] = str(log_dir / config['debug']['log_file'])
        
        # Print debug info about paths
        print(f"Log directory: {config['debug']['log_dir']}")
        print(f"Log file: {config['debug']['log_file']}")
    
    return config

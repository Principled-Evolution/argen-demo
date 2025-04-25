"""
Script to list available Predibase deployments.
"""

import json
import os
import predibase as pb

def get_api_token():
    """
    Get the Predibase API token from the config file.
    
    Returns:
        str: The API token
    """
    config_path = os.path.expanduser("~/.predibase/config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get("api_key")
    
    return None

def main():
    """List available Predibase deployments."""
    # Get API token
    api_token = get_api_token()
    
    if not api_token:
        print("Error: API token not found in ~/.predibase/config.json")
        return
    
    # Initialize Predibase client
    pb_client = pb.Predibase(api_token=api_token)
    
    # List deployments
    print("Available Predibase deployments:")
    deployments = pb_client.deployments.list()
    
    for deployment in deployments:
        print(f"- {deployment.name}")

if __name__ == "__main__":
    main()

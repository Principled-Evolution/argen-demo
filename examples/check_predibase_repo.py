"""
Script to check and create a repository in Predibase.
"""

import sys
import os
import json
import argparse
from typing import Optional

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_predibase_client():
    """
    Get a Predibase client.
    
    Returns:
        Predibase client
    """
    try:
        from predibase import Predibase
    except ImportError:
        raise ImportError(
            "Predibase SDK not installed. Please install it with 'pip install predibase'."
        )
    
    # Get API token from config file
    config_path = os.path.expanduser("~/.predibase/config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            pb_config = json.load(f)
            api_token = pb_config.get("api_key")
            if not api_token:
                raise ValueError("API token not found in config file")
    else:
        raise ValueError("Config file not found")
    
    # Initialize Predibase client
    print(f"Initializing Predibase client...")
    pb = Predibase(api_token=api_token)
    
    return pb


def check_repo(repo_name: str) -> Optional[str]:
    """
    Check if a repository exists in Predibase.
    
    Args:
        repo_name: Name of the repository
        
    Returns:
        Repository ID if the repository exists, None otherwise
    """
    pb = get_predibase_client()
    
    # Try to get the repository
    try:
        # Check if the repos attribute has a get method
        if hasattr(pb.repos, 'get'):
            repo = pb.repos.get(repo_name)
            print(f"Repository '{repo_name}' exists with ID: {repo.id}")
            return repo.id
        else:
            print("Warning: pb.repos does not have a 'get' method. Checking available methods...")
            print(f"Available methods for repos: {dir(pb.repos)}")
            return None
    except Exception as e:
        print(f"Repository '{repo_name}' does not exist or cannot be accessed: {e}")
        return None


def create_repo(repo_name: str, description: str = "Repository for ArGen GRPO fine-tuning") -> Optional[str]:
    """
    Create a repository in Predibase.
    
    Args:
        repo_name: Name of the repository
        description: Description of the repository
        
    Returns:
        Repository ID if the repository was created successfully, None otherwise
    """
    pb = get_predibase_client()
    
    # Try to create the repository
    try:
        # Check if the repos attribute has a create method
        if hasattr(pb.repos, 'create'):
            repo = pb.repos.create(name=repo_name, description=description)
            print(f"Repository '{repo_name}' created with ID: {repo.id}")
            return repo.id
        else:
            print("Warning: pb.repos does not have a 'create' method. Checking available methods...")
            print(f"Available methods for repos: {dir(pb.repos)}")
            return None
    except Exception as e:
        print(f"Error creating repository: {e}")
        return None


def main():
    """Run the repository check script."""
    parser = argparse.ArgumentParser(description="Check and create a repository in Predibase.")
    parser.add_argument("--repo-name", type=str, default="argen-gemini-opa", help="Name of the repository in Predibase")
    parser.add_argument("--description", type=str, default="Repository for ArGen GRPO fine-tuning", help="Description of the repository")
    parser.add_argument("--create", action="store_true", help="Create the repository if it doesn't exist")
    
    args = parser.parse_args()
    
    # Check if the repository exists
    repo_id = check_repo(args.repo_name)
    
    if repo_id:
        print(f"Repository '{args.repo_name}' already exists with ID: {repo_id}")
    elif args.create:
        # Create the repository
        repo_id = create_repo(args.repo_name, args.description)
        
        if repo_id:
            print(f"Repository '{args.repo_name}' created with ID: {repo_id}")
        else:
            print(f"Failed to create repository '{args.repo_name}'")
    else:
        print(f"Repository '{args.repo_name}' does not exist. Use --create to create it.")


if __name__ == "__main__":
    main()

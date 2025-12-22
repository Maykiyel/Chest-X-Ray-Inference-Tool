#!/usr/bin/env python3
"""
Complete setup script for Chest X-Ray Inference Tool.
Creates virtual environment, installs PyTorch (CPU/GPU), and all dependencies.

Usage: python install_pytorch.py
"""

import subprocess
import sys
import platform
import os
from pathlib import Path

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def print_step(step_num, total_steps, text):
    """Print a formatted step."""
    print(f"\n[{step_num}/{total_steps}] {text}")
    print("-" * 70)

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Error: Python 3.8 or higher required. You have Python {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(
            ['nvidia-smi'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def get_cuda_version():
    """Get CUDA version from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi'], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        
        if result.returncode == 0:
            import re
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    match = re.search(r'CUDA Version: (\d+\.\d+)', line)
                    if match:
                        return float(match.group(1))
        return None
    except Exception:
        return None

def create_venv(venv_path):
    """Create virtual environment."""
    print(f"Creating virtual environment at: {venv_path}")
    
    try:
        subprocess.run(
            [sys.executable, '-m', 'venv', venv_path],
            check=True
        )
        print(f"✓ Virtual environment created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_venv_python(venv_path):
    """Get path to Python executable in virtual environment."""
    system = platform.system()
    
    if system == "Windows":
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:  # Linux, macOS
        return os.path.join(venv_path, "bin", "python")

def get_venv_pip(venv_path):
    """Get path to pip executable in virtual environment."""
    system = platform.system()
    
    if system == "Windows":
        return os.path.join(venv_path, "Scripts", "pip.exe")
    else:  # Linux, macOS
        return os.path.join(venv_path, "bin", "pip")

def upgrade_pip(venv_python):
    """Upgrade pip to latest version."""
    print("Upgrading pip to latest version...")
    
    try:
        subprocess.run(
            [venv_python, '-m', 'pip', 'install', '--upgrade', 'pip'],
            check=True
        )
        print("✓ pip upgraded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠ Warning: Failed to upgrade pip: {e}")
        return False

def install_pytorch(venv_pip, use_gpu=False, cuda_version=None):
    """Install PyTorch with appropriate CUDA support."""
    
    if use_gpu:
        print(f"\n✓ NVIDIA GPU detected!")
        print(f"✓ CUDA Version: {cuda_version}")
        
        # Determine which CUDA version to install
        if cuda_version >= 12.1:
            torch_index = "cu121"
            print("→ Installing PyTorch with CUDA 12.1 support...")
        elif cuda_version >= 11.8:
            torch_index = "cu118"
            print("→ Installing PyTorch with CUDA 11.8 support...")
        else:
            torch_index = "cu117"
            print("→ Installing PyTorch with CUDA 11.7 support...")
        
        install_cmd = [
            venv_pip, 'install',
            'torch>=2.0.0',
            'torchvision>=0.15.0',
            '--index-url', f'https://download.pytorch.org/whl/{torch_index}'
        ]
    else:
        print("\n⚠ No NVIDIA GPU detected")
        print("→ Installing CPU-only PyTorch...")
        
        install_cmd = [
            venv_pip, 'install',
            'torch>=2.0.0',
            'torchvision>=0.15.0'
        ]
    
    print("\nInstalling PyTorch (this may take 5-10 minutes)...")
    print("Please be patient, downloading ~2GB of data...\n")
    
    try:
        result = subprocess.run(install_cmd, check=True)
        print("\n✓ PyTorch installation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ PyTorch installation failed: {e}")
        return False

def install_requirements(venv_pip, requirements_file='requirements.txt'):
    """Install remaining requirements from requirements.txt."""
    
    if not os.path.exists(requirements_file):
        print(f"⚠ Warning: {requirements_file} not found, skipping...")
        return False
    
    print(f"\nInstalling remaining dependencies from {requirements_file}...")
    print("This may take 3-5 minutes...\n")
    
    try:
        subprocess.run(
            [venv_pip, 'install', '-r', requirements_file],
            check=True
        )
        print("\n✓ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to install requirements: {e}")
        return False

def verify_installation(venv_python):
    """Verify PyTorch and CUDA installation."""
    print("\nVerifying installation...")
    
    verify_cmd = [
        venv_python, '-c',
        "import torch; import torchxrayvision; "
        "print(f'PyTorch: {torch.__version__}'); "
        "print(f'CUDA Available: {torch.cuda.is_available()}'); "
        "print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); "
        "print(f'TorchXRayVision: Installed'); "
        "print('\\n✓ All core packages verified!')"
    ]
    
    try:
        result = subprocess.run(verify_cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Verification failed: {e}")
        return False

def get_activation_command(venv_path):
    """Get the command to activate virtual environment."""
    system = platform.system()
    
    if system == "Windows":
        return f"{venv_path}\\Scripts\\activate"
    else:  # Linux, macOS
        return f"source {venv_path}/bin/activate"

def print_next_steps(venv_path):
    """Print instructions for next steps."""
    system = platform.system()
    activation_cmd = get_activation_command(venv_path)
    
    print_header("🎉 INSTALLATION COMPLETE!")
    
    print("Next steps:\n")
    
    print("1. Activate the virtual environment:")
    if system == "Windows":
        print(f"   Command Prompt:  {venv_path}\\Scripts\\activate.bat")
        print(f"   PowerShell:      {venv_path}\\Scripts\\Activate.ps1")
    else:
        print(f"   {activation_cmd}")
    
    print("\n2. Run the application:")
    print("   streamlit run app.py")
    
    print("\n3. Open your browser to:")
    print("   http://localhost:8501")
    
    print("\n" + "-" * 70)
    print("\n💡 Tips:")
    print("   - The virtual environment must be activated each time you use the app")
    print("   - To deactivate: type 'deactivate' in the terminal")
    print("   - To reactivate later: run the activation command above")
    
    if system == "Windows":
        print("\n⚠ Windows PowerShell users:")
        print("   If you get an execution policy error, run:")
        print("   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser")
    
    print("\n" + "=" * 70 + "\n")

def main():
    """Main installation function."""
    
    print_header("Chest X-Ray Inference Tool - Complete Setup")
    
    # Configuration
    venv_name = "xray_env"
    venv_path = os.path.join(os.getcwd(), venv_name)
    total_steps = 6
    
    # Step 1: Check Python version
    print_step(1, total_steps, "Checking Python version")
    check_python_version()
    print(f"Operating System: {platform.system()}")
    print(f"Python Executable: {sys.executable}")
    
    # Step 2: Detect GPU
    print_step(2, total_steps, "Detecting GPU capabilities")
    has_gpu = check_nvidia_gpu()
    cuda_version = get_cuda_version() if has_gpu else None
    
    if has_gpu:
        print(f"✓ NVIDIA GPU detected!")
        print(f"✓ CUDA Version: {cuda_version}")
    else:
        print("ℹ No NVIDIA GPU detected")
        print("  Will install CPU-only version")
    
    # Ask user for GPU preference
    if has_gpu:
        print("\n" + "-" * 70)
        response = input("\nInstall GPU-accelerated PyTorch? (Y/n): ").strip().lower()
        use_gpu = response != 'n'
        if not use_gpu:
            print("→ User selected CPU-only installation")
    else:
        use_gpu = False
    
    # Step 3: Create virtual environment
    print_step(3, total_steps, "Creating virtual environment")
    
    if os.path.exists(venv_path):
        print(f"⚠ Virtual environment already exists at: {venv_path}")
        response = input("Delete and recreate? (y/N): ").strip().lower()
        
        if response == 'y':
            print("Removing existing virtual environment...")
            import shutil
            shutil.rmtree(venv_path)
            print("✓ Removed")
        else:
            print("Using existing virtual environment...")
    
    if not os.path.exists(venv_path):
        if not create_venv(venv_path):
            print("\n❌ Setup failed at virtual environment creation")
            sys.exit(1)
    
    # Get venv executables
    venv_python = get_venv_python(venv_path)
    venv_pip = get_venv_pip(venv_path)
    
    print(f"✓ Virtual environment ready at: {venv_path}")
    print(f"✓ Python: {venv_python}")
    print(f"✓ Pip: {venv_pip}")
    
    # Step 4: Upgrade pip
    print_step(4, total_steps, "Upgrading pip")
    upgrade_pip(venv_python)
    
    # Step 5: Install PyTorch
    print_step(5, total_steps, "Installing PyTorch")
    if not install_pytorch(venv_pip, use_gpu, cuda_version):
        print("\n❌ Setup failed at PyTorch installation")
        sys.exit(1)
    
    # Step 6: Install remaining requirements
    print_step(6, total_steps, "Installing remaining dependencies")
    if not install_requirements(venv_pip):
        print("\n⚠ Warning: Some dependencies may not have been installed")
    
    # Verify installation
    print_header("Verifying Installation")
    if not verify_installation(venv_python):
        print("\n⚠ Warning: Verification failed, but you can try running the app")
    
    # Print next steps
    print_next_steps(venv_path)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
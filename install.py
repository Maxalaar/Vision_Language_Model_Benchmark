from utilities import install_command

if __name__ == '__main__':
    # List of commands to run
    commands = [
        'pip install torch torchvision',
        'pip install transformers',
        'pip install accelerate',
        'pip install -U vllm',
        'pip install qwen-vl-utils==0.0.14',
    ]

    # Execute conda commands
    for command in commands:
        install_command(command)

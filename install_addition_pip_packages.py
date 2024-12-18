import yaml

def install_additional_pip_packages(
        existing_packages_path: str, 
        added_environment_path: str,
        out_path: str
):
    '''
    Reference: 
        https://stackoverflow.com/questions/72824468/pip-installing-environment-yml-as-if-its-a-requirements-txt
    '''

    with open(added_environment_path) as file_handle:
        environment_data = yaml.safe_load(file_handle)

    # Load existing packages
    with open(existing_packages_path) as f:
        current_packages = dict(line.strip().split('==') for line in f if '==' in line)

    out_str = ""
    for dependency in environment_data["dependencies"]:
        if isinstance(dependency, dict):
            for lib in dependency['pip']:
                new_package = lib.split('==')[0]
                current_version = current_packages.get(new_package)
                if current_version:
                    has_version = len(lib.split('==')) > 0
                    if has_version:
                        required_version = lib.split('==')[1]
                        if current_version != required_version:
                            print(f"Package {new_package} is currently at version {current_version}, but requires {required_version}.")
                            user_input = input("Do you want to update this package? (yes/no): ")
                            if user_input.lower() == 'yes':
                                out_str += lib + "\n"
                else:
                    # Install new package if it's not present
                    out_str += lib + "\n"   

    with open(out_path, "w") as f:
        f.write(out_str)

if __name__ == "__main__":
    install_additional_pip_packages(
        existing_packages_path="current_pip_packages.txt",
        added_environment_path="environment.yaml",
        out_path="requirements_updated.txt"
    )


        
        


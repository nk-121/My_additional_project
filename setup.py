import os

BASE_DIR = "backend"

structure = {
    "app": {
        "__init__.py": "",
        "main.py": "",
        "controllers": {
            "__init__.py": "",
            "customer_controller.py": ""
        },
        "models": {
            "__init__.py": "",
            "customer.py": ""
        },
        "schemas": {
            "__init__.py": "",
            "customer_schema.py": ""
        },
        "db": {
            "__init__.py": "",
            "database.py": ""
        },
        "utils": {
            "__init__.py": "",
            "response_wrapper.py": ""
        }
    },
    "requirements.txt": ""
}


def create_structure(base_path, structure_dict):
    for name, content in structure_dict.items():
        path = os.path.join(base_path, name)

        # If it's a file
        if isinstance(content, str):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)

        # If it's a directory
        elif isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)


if __name__ == "__main__":
    create_structure(".", {BASE_DIR: structure})
    print("✅ FastAPI MVC structure created successfully!")

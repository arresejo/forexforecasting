#!/bin/bash

script_dir=$(dirname "$0")

clean_folder() {
    folder_name="$1"
    read -p "Do you want to clean the directory '$folder_name'? [Y/n] " response
    response=$(echo "$response" | tr '[:upper:]' '[:lower:]')
    if [[ -z "$response" || "$response" == "y" ]]; then
        if [ -d "$folder_name" ]; then
            rm -rf "$folder_name"/*
            echo "All contents within the directory '$folder_name' have been removed."
        else
            echo "Directory '$folder_name' does not exist."
        fi
    else
        echo "Skipping cleaning of directory '$folder_name'."
    fi
}

clean_folder "$script_dir/../processed"
clean_folder "$script_dir/../reports"

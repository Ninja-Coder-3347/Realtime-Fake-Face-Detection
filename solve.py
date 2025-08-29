import os

def fix_labels(label_folder):
    for file_name in os.listdir(label_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(label_folder, file_name)
            with open(file_path, "r") as f:
                lines = f.readlines()

            fixed_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 1:  # Likely a concatenated string of numbers
                    # Attempt to split based on digit sequences
                    fixed_line = " ".join(filter(lambda x: x, parts[0].split("0.")))
                    fixed_line = f"0.{fixed_line}".replace(" ", " 0.")
                    fixed_lines.append(fixed_line + "\n")
                elif len(parts) == 5:
                    fixed_lines.append(" ".join(parts) + "\n")

            # Write the fixed lines back to the file
            with open(file_path, "w") as f:
                f.writelines(fixed_lines)

# Paths to train and validation label folders
train_label_folder = r"C:\Users\pallavi\PycharmProjects\facedetect\Dataset\SplitData\train\labels"
val_label_folder = r"C:\Users\pallavi\PycharmProjects\facedetect\Dataset\SplitData\val\labels"

fix_labels(train_label_folder)
fix_labels(val_label_folder)
print("Label files have been fixed!")

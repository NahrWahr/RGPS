import re

def parse_rinex_header(file_path):
    """
    Parse the header of a RINEX v3 OBS file.
    Returns a dictionary where keys are the header labels and values are the corresponding text.
    """
    header = {}
    header_lines = []
    with open(file_path, 'r') as f:
        for line in f:
            # Each header line has the last 20 characters as the header label.
            label = line[60:80].strip()
            header_lines.append(line.rstrip())
            if label == "END OF HEADER":
                break
            # For many header lines, the useful value is in the first 60 characters.
            value = line[:60].strip()
            # Store the line by its label; if multiple lines exist with the same label, append to a list.
            if label in header:
                if isinstance(header[label], list):
                    header[label].append(value)
                else:
                    header[label] = [header[label], value]
            else:
                header[label] = value
    return header, header_lines

def parse_first_epoch(file_path):
    """
    After the header, parse the first observation epoch.
    This function assumes the file follows the header with an epoch record.
    RINEX v3 observation epochs start with a line beginning with '>'.
    """
    with open(file_path, 'r') as f:
        # Skip header (until END OF HEADER)
        for line in f:
            if line[60:80].strip() == "END OF HEADER":
                break
        # Now find the first epoch line.
        for line in f:
            if line.startswith('>'):
                # Example epoch line (fields are fixed-width):
                # > 2019 01 25 00 16 30.0000000  0 12
                parts = line[1:80].split()
                if len(parts) >= 7:
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    hour = int(parts[3])
                    minute = int(parts[4])
                    second = float(parts[5])
                    epoch_flag = int(parts[6])
                    # Number of satellites follows after flag (if available)
                    nsat = int(parts[7]) if len(parts) > 7 else None
                    epoch = {
                        "year": year,
                        "month": month,
                        "day": day,
                        "hour": hour,
                        "minute": minute,
                        "second": second,
                        "epoch_flag": epoch_flag,
                        "nsat": nsat
                    }
                    return epoch
    return None

if __name__ == "__main__":
    # Replace with the path to your RINEX OBS file.
    rinex_file = "path/to/your_file.obs"
    
    # Parse the header.
    header, header_lines = parse_rinex_header(rinex_file)
    
    print("Parsed RINEX Header:")
    for key, value in header.items():
        print(f"{key}: {value}")
    
    # Parse the first observation epoch.
    epoch = parse_first_epoch(rinex_file)
    if epoch:
        print("\nFirst Observation Epoch:")
        print(epoch)
    else:
        print("No observation epoch found.")

    # You can extend the parser to read subsequent observation data records.

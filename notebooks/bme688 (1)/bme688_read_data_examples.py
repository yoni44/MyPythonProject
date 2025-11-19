import json

def bme688_read_bmerawdata_file(file_path):
    """
    Reads and parses a .bmerawdata file containing JSON data.

    Args:
        file_path (str): Path to the .bmerawdata file.

    Returns:
        dict: Parsed JSON data as a Python dictionary.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON. Details: {e}")
        return None

def bme688_initialize_data_buffer(config_body):
    """
    Initializes a nested dictionary structure for the data buffer, dynamically adding profiles.

    Args:
        config_body (dict): Configuration body containing sensor configurations.

    Returns:
        dict: Initialized data buffer.
    """
    
    sensors_config = config_body.get("sensorConfigurations", [])
    heater_profiles = config_body.get("heaterProfiles", [])
    
    data_buffer = {}
    profiles_sensors = {}
    

    for sensor_config in sensors_config:
        profile_name = sensor_config.get("heaterProfile", str)
        sensor_idx = sensor_config.get("sensorIndex", int)
        if profile_name not in profiles_sensors:
            profiles_sensors[profile_name] = {"sensors_idx": [], "heat_steps": int}
            profiles_sensors[profile_name]["sensors_idx"].append(sensor_idx)

            # Add heat steps for heater profile
            for profile in heater_profiles:
                if profile_name == profile["id"]:
                    profiles_sensors[profile_name]["heat_steps"] = len(profile["temperatureTimeVectors"])

        else:
            profiles_sensors[profile_name]["sensors_idx"].append(sensor_idx)

    for profile_name in profiles_sensors:        
        if profile_name not in data_buffer:
            data_buffer[profile_name] = {
                "sensors": [
                    {
                        "sensor_idx": profiles_sensors[profile_name]["sensors_idx"][sensor_index],
                        "heater_steps": [
                            {
                                "heat_step_idx": heat_step_idx,
                                "temperature": [],
                                "pressure": [],
                                "humidity": [],
                                "resistance": []
                            }
                            for heat_step_idx in range((profiles_sensors[profile_name]["heat_steps"]))  # 0-10 indexes
                        ]
                    }
                    for sensor_index in range(len(profiles_sensors[profile_name]["sensors_idx"]))  # 0-7 indexes
                ]
            }

    return data_buffer

def bme688_fill_bmerawdata_data_buffer(data_buffer, data_block):
        
    for measure in data_block:
        for profile in data_buffer:
            for sensor in data_buffer[profile]["sensors"]:
                if measure[0] == sensor["sensor_idx"]:
                    for heat_step in sensor["heater_steps"]:
                        if measure[8] == heat_step["heat_step_idx"]:
                            heat_step["temperature"].append(measure[4])
                            heat_step["pressure"].append(measure[5])
                            heat_step["humidity"].append(measure[6])
                            heat_step["resistance"].append(measure[7])
    
    return data_buffer




    """
    Initializes a data buffer for other sensors.

    Returns:
        dict: Initialized data buffer.
    """
    
    # zphs01b sensor
    zphs01b_fields = ["voc", "temp", "pm2_5", "pm1_0", "pm10", "o3", "no2", "hum", "fmhd", "co2", "co"]
    
    # sfa30 sensor
    sfa30_fields = ["temp", "hum", "fmhd"]
    
    # sen5x sensor
    sen5x_fields = ["voc", "temp", "pm4_0", "pm2_5", "pm1_0", "pm1_0", "pm10", "nox", "mass_c", "hum"]

    # scd41 sensor
    scd41_fields = ["co2", "temp", "hum"]

    # mics6814 sensor
    mics6814_fields = ["red_res", "no2_res", "no2", "nh3_res", "nh3", "co_res", "co"]

    # buffer initialization
    data_buffer = {
        "ts": [],
        "zphs01b": {field: [] for field in zphs01b_fields},
        "sfa30": {field: [] for field in sfa30_fields},
        "sen5x": {field: [] for field in sen5x_fields},
        "scd41": {field: [] for field in scd41_fields},
        "mics6814": {field: [] for field in mics6814_fields}
    }
    return data_buffer




# Example of usage
if __name__ == "__main__":
    
    data_type = "bmerawdata"
    
    if data_type == "bmerawdata":
        
        file_path = "2025_02_11_12_16_Board_4022D8F3CCF0_PowerOnOff_1_15bkfr4c0luuhp84_File_1.bmerawdata"  # Replace with your file's path
        
        data = bme688_read_bmerawdata_file(file_path)

        if data:
            print("Successfully parsed the JSON data!")

            # Accessing different sections of the data
            #config_header = data.get("configHeader", {})
            config_body = data.get("configBody", {})
            #raw_data_header = data.get("rawDataHeader", {})
            raw_data_body = data.get("rawDataBody", {})
            
            #sensors_config = config_body.get("sensorConfigurations", [])
            data_block = raw_data_body.get("dataBlock", [])
            

            # Example: Accessing heater profiles
            heater_profiles = config_body.get("heaterProfiles", [])
            print("Heater Profiles:", heater_profiles)

            # Example: Accessing data columns
            data_columns = raw_data_body.get("dataColumns", [])
            print("Data Columns:", data_columns)


            # Create structured data cantainer for ML
            data_structured_init = bme688_initialize_data_buffer(config_body)
            data_structured_filled = bme688_fill_bmerawdata_data_buffer(data_structured_init, data_block)

            print("", data_structured_filled)

        else:
            print("Failed to read or parse the file.")
    
        
    else:
        print("Unsupported data type. Please choose 'bmerawdata'.")
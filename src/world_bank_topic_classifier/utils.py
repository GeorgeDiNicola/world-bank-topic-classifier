
import os

def check_env_boolean_required(flag_name: str) -> bool:
    flag_value = os.getenv(flag_name).lower().strip()

    if flag_value in ("true", "1", "yes", "t"):
        print(f"Environment variable '{flag_name}' is True.")
        return True
    elif flag_value in ("false", "0", "no", "f"):
        print(f"Environment variable '{flag_name}' is False.")
        return False
    else:
        raise ValueError(
            f"Environment variable '{flag_name}' exists but contains an invalid boolean string: '{flag_value}'"
        )

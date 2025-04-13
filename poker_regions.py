# Definiciones de regiones para la mesa de póker

# Región principal de la mesa de póker
MANUAL_GAME_REGION = {
    "top": 526,
    "left": 247,
    "width": 711,
    "height": 732
}

# Sub-regiones dentro de la mesa de póker
sub_regions = {
    "community_cards": {
        "top": 624,
        "left": 270,
        "width": 669,
        "height": 177
    },
    "player_cards": {
        "top": 821,
        "left": 465,
        "width": 273,
        "height": 178
    },
    "pot": {
        "top": 529,
        "left": 479,
        "width": 245,
        "height": 69
    },
    "player_balance": {
        "top": 1006,
        "left": 555,
        "width": 101,
        "height": 33
    },
    "fold_button": {
        "top": 1200,
        "left": 297,
        "width": 102,
        "height": 53
    },
    "check_call_button": {
        "top": 1196,
        "left": 523,
        "width": 173,
        "height": 57
    },
    "raise_button": {
        "top": 1204,
        "left": 804,
        "width": 115,
        "height": 42
    },
}

# Función auxiliar para acceder a botones fácilmente
def get_buttons():
    return {
        "fold": {
            "top": 1200,
            "left": 297,
            "width": 102,
            "height": 53
        },
        "check_call": {
            "top": 1196,
            "left": 523,
            "width": 173,
            "height": 57
        },
        "raise": {
            "top": 1204,
            "left": 804,
            "width": 115,
            "height": 42
        },
    }

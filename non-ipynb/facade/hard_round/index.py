"""
hard_round
    Converts the float into string first, and then turn it into float by substring
    the string value according to the configured decimal_places.

    @params float32 number
    @params int decimal_place

    @returns float32
"""
def hard_round(number, decimal_places = 0):
    non_coma_value_count = 2

    number_in_string: str = str(number)

    behind_coma_value_count = len(number_in_string.split(".")[1])

    if(behind_coma_value_count > 4):
        behind_coma_value_count = 4

    if(number > 9.9):
        non_coma_value_count = len(number_in_string.split(".")[0]) + 1

    value_length = len(number_in_string)

    substring_size = decimal_places + non_coma_value_count # Don't forget the front number 0.

    if(value_length < substring_size):
        for i in range(0, (substring_size - value_length)):
            number_in_string = number_in_string + "0"

    return float(number_in_string[0:behind_coma_value_count + non_coma_value_count])
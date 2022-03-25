def calculate_number_of_rotate(location, side_match_for_the_location):
    if location == 1:
        if side_match_for_the_location == 1:
            answer = 2
        elif side_match_for_the_location == 2:
            answer = 1
        elif side_match_for_the_location == 3:
            answer = 0
        elif side_match_for_the_location == 4:
            answer = -1

    if location == 2:
        if side_match_for_the_location == 1:
            answer = -1
        elif side_match_for_the_location == 2:
            answer = 2
        elif side_match_for_the_location == 3:
            answer = 1
        elif side_match_for_the_location == 4:
            answer = 0      

    if location == 3:
        if side_match_for_the_location == 1:
            answer = 0
        elif side_match_for_the_location == 2:
            answer = -1
        elif side_match_for_the_location == 3:
            answer = 2
        elif side_match_for_the_location == 4:
            answer = 1

    if location == 4:
        if side_match_for_the_location == 1:
            answer = 1
        elif side_match_for_the_location == 2:
            answer = 0
        elif side_match_for_the_location == 3:
            answer = -1
        elif side_match_for_the_location == 4:
            answer = 2

    return answer


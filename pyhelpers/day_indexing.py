# this code should contain a script that takes in the day and sets that as index 0. Then take another day and set that as difference from today.
# for example if today is wedneday, monday should be index 3, since we're not counting weekends
def day_to_index(start_day, target_day):
    days = ['måndag', 'tisdag', 'onsdag', 'torsdag', 'fredag']
    start_index = days.index(start_day.lower())
    target_index = days.index(target_day.lower())
    if target_index >= start_index:
        return target_index - start_index
    else:
        return (target_index + 5) - start_index
    
def index_to_day(start_day, index):
    days = ['måndag', 'tisdag', 'onsdag', 'torsdag', 'fredag']
    start_index = days.index(start_day.lower())
    target_index = (start_index + index) % 5
    return days[target_index]

def weekday_name_sv(dt):
    SV_DAYS = ["måndag", "tisdag", "onsdag", "torsdag", "fredag", "lördag", "söndag"]
    return SV_DAYS[dt.weekday()]  # weekday(): Mon=0...Sun=6

def main():
    # Example usage:
    start_day = 'fredag'
    target_day = 'torsdag'
    index = day_to_index(start_day, target_day)
    print(f"The index of {target_day} when starting from {start_day} is: {index}")
    
if __name__ == "__main__":
    main()
    
    


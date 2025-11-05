# a class detailing a job order to be done by workers
class Jobclass:
    def __init__(self, job_id, description, adress, service_time, manpower_required=1, day_required = None, time_slot_required = None, periodicity = None ):
        self.job_id = job_id
        self.description = description
        self.adress = adress
        self.service_time = service_time # 
        self.manpower_required = manpower_required # integer, number of workers required, default 1
        self.day_required = day_required # monday, tuesday, etc. or None if no specific day is required
        self.time_slot_required = time_slot_required
        self.periodicity = periodicity
        
    def __repr__(self):
        return f"Job_order({self.job_id}, {self.description}, {self.adress}, {self.service_time}, {self.manpower_required}, {self.day_required}, {self.time_slot_required}, {self.periodicity})"
    
    def get_job_adress(self):
        return self.adress
# a class detailing a job order to be done by workers
class Jobclass:
    """Representation of a single job retrieved from ``September_arbete``.

    The class stores the first 16 columns from the Excel sheet along with the
    calculated ``service_time`` that is derived from the time span in the ``Tid``
    column.  Additional legacy attributes (``day_required``, ``time_slot_required``
    and ``periodicity``) are kept for backwards compatibility with older code.
    """

    def __init__(
        self,
        job_id,
        *,
        day,
        date,
        time_slot,
        break_minutes,
        staff_name,
        service_name,
        work_address,
        work_postcode,
        project_number,
        project_name,
        customer_number,
        customer_name,
        billing_reference,
        job_type,
        status,
        booking_frequency,
        service_time,
    ):
        self.job_id = job_id
        self.day = day
        self.date = date
        self.time_slot = time_slot
        self.break_minutes = break_minutes
        self.staff_name = staff_name
        self.service_name = service_name
        self.work_address = work_address
        self.work_postcode = work_postcode
        self.project_number = project_number
        self.project_name = project_name
        self.customer_number = customer_number
        self.customer_name = customer_name
        self.billing_reference = billing_reference
        self.job_type = job_type
        self.status = status
        self.booking_frequency = booking_frequency
        self.service_time = service_time

        # Fields kept for backwards compatibility with earlier iterations of the
        # repository code base.
        self.description = f"{self.service_name} {self.project_number}".strip()
        self.adress = ", ".join(
            part for part in [str(self.work_address).strip(), str(self.work_postcode).strip()]
            if part and part.lower() != "nan"
        )
        self.manpower_required = 1
        self.day_required = self.day
        self.time_slot_required = self.time_slot
        self.periodicity = self.booking_frequency

    def __repr__(self):
        return (
            "Job_order("
            f"{self.job_id}, {self.service_name}, {self.adress}, {self.service_time}, "
            f"{self.manpower_required}, {self.day_required}, {self.time_slot_required}, {self.periodicity}"
            ")"
        )

    def get_job_adress(self):
        return self.adress
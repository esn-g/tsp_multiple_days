"""Worker representation used when parsing staff details files."""


class Workerclass:
    """Representation of a worker from ``September_staff_details``."""

    def __init__(
        self,
        *,
        name,
        has_driver_license,
        employee_number,
        days_per_week,
        hours_per_week,
        employment_percentage,
        employment_terms,
        salary_type,
    ):
        self.name = name
        self.has_driver_license = has_driver_license
        self.employee_number = employee_number
        self.days_per_week = days_per_week
        self.hours_per_week = hours_per_week
        self.employment_percentage = employment_percentage
        self.employment_terms = employment_terms
        self.salary_type = salary_type

    def __repr__(self):
        return (
            "Worker("
            f"{self.name}, license={self.has_driver_license}, "
            f"days/week={self.days_per_week}, hours/week={self.hours_per_week}"
            ")"
        )


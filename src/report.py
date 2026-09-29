import pandas as pd
from datetime import datetime


def create_report(
    pcb_name,
    detections
):

    number_of_defects = len(
        detections
    )

    status = (
        "PASS"
        if number_of_defects == 0
        else "FAIL"
    )

    if detections:

        average_confidence = (
            sum(
                item["confidence"]
                for item in detections
            )
            / number_of_defects
        )

    else:

        average_confidence = 100.0

    report = {

        "PCB_ID": pcb_name,

        "Date_Time":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "Defects":
            number_of_defects,

        "Average_Confidence":
            round(
                average_confidence,
                2
            ),

        "Status":
            status
    }

    return pd.DataFrame(
        [report]
    )
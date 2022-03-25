import matplotlib.pyplot as plt
import numpy as np
import hashlib
from datetime import datetime

plt.style.use('ggplot')


class EvaluationMetric:
    def __init__(self, title, horizontal_label, vertical_label):
        self.title = title
        self.x = horizontal_label
        self.y = vertical_label
        self.vertical_values = []
        self.horizonal_values = []

    def add(self, v_value, h_value):
        """
        Adds a value pair to the evaluation metric.

        Parameters
        ----------
        v_value The value for the vertical axis
        h_value The value for the horizonal axis

        Returns
        -------
        None
        """

        self.vertical_values.append(v_value)
        self.horizonal_values.append(h_value)

    def create_plot(self, report_location):
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(self.horizonal_values, self.vertical_values)

        ax.set_title(self.title)
        ax.set_xlabel(self.x, fontsize=14)
        ax.set_ylabel(self.y, fontsize=14)
        # Create the save file name
        # use the current time as the seed
        now = datetime.now()
        # use the date hash as the file name
        date_hash = hashlib.md5(str(now).encode())
        image_name = date_hash.hexdigest()

        full_name = f"{report_location}images/{image_name}.png"
        plt.savefig(full_name)

        return f"images/{image_name}.png"

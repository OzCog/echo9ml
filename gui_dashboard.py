import tkinter as tk
from tkinter import ttk, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkintertable import TableCanvas, TableModel
from activity_regulation import ActivityRegulator
import threading
import psutil
import os
import json
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GUIDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep Tree Echo Dashboard")
        self.root.geometry("800x600")

        self.activity_regulator = ActivityRegulator()
        self.activity_thread = threading.Thread(target=self.activity_regulator.run, daemon=True)
        self.activity_thread.start()

        self.create_widgets()
        self.update_system_health()
        self.update_activity_logs()

    def create_widgets(self):
        self.tab_control = ttk.Notebook(self.root)

        self.dashboard_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.dashboard_tab, text="Dashboard")

        self.system_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.system_tab, text="System Health")

        self.logs_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.logs_tab, text="Activity Logs")

        self.tasks_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tasks_tab, text="Task Management")

        self.tab_control.pack(expand=1, fill="both")

        self.create_dashboard_tab()
        self.create_system_tab()
        self.create_logs_tab()
        self.create_tasks_tab()

    def create_dashboard_tab(self):
        self.dashboard_frame = ttk.Frame(self.dashboard_tab)
        self.dashboard_frame.pack(expand=1, fill="both")

        self.summary_label = ttk.Label(self.dashboard_frame, text="System Summary")
        self.summary_label.pack(pady=10)

        self.summary_text = tk.Text(self.dashboard_frame, wrap="word", height=10)
        self.summary_text.pack(expand=1, fill="both")

        self.pie_chart_frame = ttk.Frame(self.dashboard_frame)
        self.pie_chart_frame.pack(expand=1, fill="both")

        self.update_dashboard()

    def create_system_tab(self):
        self.cpu_label = ttk.Label(self.system_tab, text="CPU Usage: ")
        self.cpu_label.pack(pady=10)

        self.memory_label = ttk.Label(self.system_tab, text="Memory Usage: ")
        self.memory_label.pack(pady=10)

        self.disk_label = ttk.Label(self.system_tab, text="Disk Usage: ")
        self.disk_label.pack(pady=10)

    def create_logs_tab(self):
        self.logs_text = tk.Text(self.logs_tab, wrap="word")
        self.logs_text.pack(expand=1, fill="both")

        self.search_entry = ttk.Entry(self.logs_tab)
        self.search_entry.pack(pady=10)

        self.search_button = ttk.Button(self.logs_tab, text="Search Logs", command=self.search_logs)
        self.search_button.pack(pady=10)

    def create_tasks_tab(self):
        self.task_listbox = tk.Listbox(self.tasks_tab)
        self.task_listbox.pack(expand=1, fill="both")

        self.add_task_entry = ttk.Entry(self.tasks_tab)
        self.add_task_entry.pack(pady=10)

        self.add_task_button = ttk.Button(self.tasks_tab, text="Add Task", command=self.add_task)
        self.add_task_button.pack(pady=10)

        self.remove_task_button = ttk.Button(self.tasks_tab, text="Remove Task", command=self.remove_task)
        self.remove_task_button.pack(pady=10)

        self.prioritize_task_button = ttk.Button(self.tasks_tab, text="Prioritize Task", command=self.prioritize_task)
        self.prioritize_task_button.pack(pady=10)

    def update_dashboard(self):
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        summary = f"CPU Usage: {cpu_usage}%\nMemory Usage: {memory.percent}%\nDisk Usage: {disk.percent}%"
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, summary)

        self.create_pie_chart(cpu_usage, memory.percent, disk.percent)

        self.root.after(1000, self.update_dashboard)

    def create_pie_chart(self, cpu, memory, disk):
        fig, ax = plt.subplots()
        labels = 'CPU', 'Memory', 'Disk'
        sizes = [cpu, memory, disk]
        colors = ['gold', 'yellowgreen', 'lightcoral']
        explode = (0.1, 0, 0)

        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
        ax.axis('equal')

        for widget in self.pie_chart_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.pie_chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=1, fill="both")

    def update_system_health(self):
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        self.cpu_label.config(text=f"CPU Usage: {cpu_usage}%")
        self.memory_label.config(text=f"Memory Usage: {memory.percent}%")
        self.disk_label.config(text=f"Disk Usage: {disk.percent}%")

        self.root.after(1000, self.update_system_health)

    def update_activity_logs(self):
        logs_dir = Path('activity_logs')
        logs = []
        for component in logs_dir.iterdir():
            if component.is_dir():
                activity_file = component / 'activity.json'
                if activity_file.exists():
                    with open(activity_file) as f:
                        logs.extend(json.load(f))

        self.logs_text.delete(1.0, tk.END)
        for log in logs:
            self.logs_text.insert(tk.END, f"{log['time']}: {log['description']}\n")

        self.root.after(5000, self.update_activity_logs)

    def search_logs(self):
        search_term = self.search_entry.get()
        if search_term:
            logs_dir = Path('activity_logs')
            logs = []
            for component in logs_dir.iterdir():
                if component.is_dir():
                    activity_file = component / 'activity.json'
                    if activity_file.exists():
                        with open(activity_file) as f:
                            logs.extend(json.load(f))

            self.logs_text.delete(1.0, tk.END)
            for log in logs:
                if search_term.lower() in log['description'].lower():
                    self.logs_text.insert(tk.END, f"{log['time']}: {log['description']}\n")

    def add_task(self):
        task_id = self.add_task_entry.get()
        if task_id:
            self.activity_regulator.add_task(task_id, lambda: print(f"Executing {task_id}"))
            self.task_listbox.insert(tk.END, task_id)
            self.add_task_entry.delete(0, tk.END)

    def remove_task(self):
        selected_task = self.task_listbox.curselection()
        if selected_task:
            task_id = self.task_listbox.get(selected_task)
            self.activity_regulator.remove_task(task_id)
            self.task_listbox.delete(selected_task)

    def prioritize_task(self):
        selected_task = self.task_listbox.curselection()
        if selected_task:
            task_id = self.task_listbox.get(selected_task)
            self.activity_regulator.prioritize_task(task_id)
            messagebox.showinfo("Task Prioritized", f"Task {task_id} has been prioritized.")

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = GUIDashboard(root)
    root.mainloop()

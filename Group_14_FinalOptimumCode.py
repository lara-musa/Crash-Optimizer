import itertools
import numpy as np
from typing import List, Dict, Any, Tuple
import random
import math

class Task:
    def __init__(self, task_id: str, methods: list, predecessors: list = None):
        self.id = task_id
        self.methods = methods
        self.predecessors = predecessors or []
        
        # Calculation attributes
        self.early_start = 0
        self.early_finish = 0
        self.late_start = 0
        self.late_finish = 0
        self.total_float = 0
        self.current_method = None
        self.current_duration = None

class SimulatedAnnealingProjectOptimizer:
    def __init__(self, tasks, max_project_duration, budget, bonus_schedule, 
                 initial_temperature=1000, cooling_rate=0.95, 
                 iterations=1000):
        """
        Initialize the Simulated Annealing Project Optimizer
        
        :param tasks: List of tasks in the project
        :param max_project_duration: Maximum allowed project duration
        :param budget: Total budget for crashing
        :param bonus_schedule: Bonus for early completion
        :param initial_temperature: Starting temperature for simulated annealing
        :param cooling_rate: Rate at which temperature decreases
        :param iterations: Number of iterations to run
        """
        self.tasks = {task.id: task for task in tasks}
        self.max_project_duration = max_project_duration
        self.budget = budget
        self.bonus_schedule = bonus_schedule
        
        # Simulated Annealing parameters
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.iterations = iterations
    
    def _calculate_network_times(self):
            """
            Calculate early and late times for project network
            
            :return: Project duration
            """
            # Reset task times
            for task in self.tasks.values():
                task.early_start = 0
                task.early_finish = 0
                task.late_start = 0
                task.late_finish = 0
                task.total_float = 0
            
            # Forward pass (early times)
            sorted_tasks = self._topological_sort()
            for task_id in sorted_tasks:
                task = self.tasks[task_id]
                
                # Determine earliest possible start time
                if not task.predecessors:
                    task.early_start = 0
                else:
                    task.early_start = max(
                        self.tasks[pred].early_finish 
                        for pred in task.predecessors
                    )
                
                # Use current method's duration
                task.early_finish = task.early_start + task.current_duration
            
            # Backward pass (late times)
            project_duration = max(task.early_finish for task in self.tasks.values())
            sorted_tasks.reverse()
            for task_id in sorted_tasks:
                task = self.tasks[task_id]
                
                # Determine latest possible finish time
                successors = self._get_successors(task_id)
                if not successors:
                    task.late_finish = project_duration
                else:
                    task.late_finish = min(
                        self.tasks[succ].late_start 
                        for succ in successors
                    )
                
                # Calculate late start
                task.late_start = task.late_finish - task.current_duration
            
            return project_duration
    
    def _topological_sort(self):
        """
        Perform topological sorting of tasks
        
        :return: List of task IDs in topological order
        """
        in_degree = {task_id: 0 for task_id in self.tasks}
        for task in self.tasks.values():
            for pred in task.predecessors:
                in_degree[task.id] += 1
            
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        sorted_tasks = []
            
        while queue:
            current_task = queue.pop(0)
            sorted_tasks.append(current_task)
                
            for succ_id in self._get_successors(current_task):
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)
            
        return sorted_tasks
    
    def _get_successors(self, task_id):
        """
        Find successor tasks for a given task
        
        :param task_id: ID of the task
        :return: List of successor task IDs
        """
        return [
            other_task_id 
            for other_task_id, other_task in self.tasks.items() 
            if task_id in other_task.predecessors
        ]
    
    def _calculate_bonus(self, project_time):
        """
        Calculate bonus based on early completion
        
        :param project_time: Total project time
        :return: Bonus amount
        """
        max_bonus = 0
        for days_early, bonus in sorted(self.bonus_schedule.items(), reverse=True):
            if self.max_project_duration - project_time >= days_early:
                max_bonus = bonus
                break
        return max_bonus
    
    def _generate_initial_solution(self):
        """
        Generate an initial solution
        
        :return: Tuple of (method_selection, crash_option)
        """
        # Randomly select a method for each task
        method_selection = [
            random.randint(0, len(task.methods) - 1) 
            for task in self.tasks.values()
        ]
        
        # Randomly generate crash options
        crash_option = [
            random.randint(0, task.methods[method_index].get('max_crash_duration', 0))
            for task, method_index in zip(self.tasks.values(), method_selection)
        ]
        
        return method_selection, crash_option

    def _calculate_total_cost(self, method_selection, crash_option):
        """
        Calculate total project cost
        
        :param method_selection: Selected method indices
        :param crash_option: Crash days for each task
        :return: Tuple of (total_normal_cost, total_crash_cost, project_time, bonus)
        """
        # Apply selected methods and reset durations
        total_normal_cost = 0
        for task, method_index in zip(self.tasks.values(), method_selection):
            method = task.methods[method_index]
            task.current_method = method_index
            task.current_duration = method['normal_duration']
            total_normal_cost += method['normal_cost']
        
        # Apply crashing
        total_crash_cost = 0
        for (task_id, method_index), crash_days in zip(
            zip(self.tasks.keys(), method_selection), crash_option
        ):
            task = self.tasks[task_id]
            method = task.methods[method_index]
            
            # Calculate crash cost
            crash_cost_per_day = method.get('crash_cost_per_day', 0)
            total_crash_cost += crash_days * crash_cost_per_day
            
            # Reduce task duration
            task.current_duration = max(
                method['normal_duration'] - crash_days, 
                method['normal_duration'] - method.get('max_crash_duration', 0)
            )
        
        # Skip if crash cost exceeds budget
        if total_crash_cost > self.budget:
            return float('inf'), float('inf'), float('inf'), 0
        
        # Calculate project duration
        project_time = self._calculate_network_times()
        
        # Skip if project time exceeds maximum
        if project_time > self.max_project_duration:
            return float('inf'), float('inf'), float('inf'), 0
        
        # Calculate bonus
        bonus = self._calculate_bonus(project_time)
        
        return total_normal_cost, total_crash_cost, project_time, bonus
    
    def optimize(self):
        """
        Perform Simulated Annealing optimization
        
        :return: Best strategy found
        """
        # Generate initial solution
        current_method_selection, current_crash_option = self._generate_initial_solution()
        
        # Calculate initial cost
        current_normal_cost, current_crash_cost, current_project_time, current_bonus = (
            self._calculate_total_cost(current_method_selection, current_crash_option)
        )
        
        # Initialize best solution
        best_method_selection = current_method_selection
        best_crash_option = current_crash_option
        max_net_benefit = current_bonus - (current_normal_cost + current_crash_cost)
        
        # Simulated Annealing
        temperature = self.initial_temperature
        for _ in range(self.iterations):
            # Generate neighboring solution
            neighbor_method_selection = current_method_selection.copy()
            neighbor_crash_option = current_crash_option.copy()
            
            # Randomly mutate method selection
            method_index = random.randint(0, len(neighbor_method_selection) - 1)
            neighbor_method_selection[method_index] = random.randint(
                0, len(self.tasks[list(self.tasks.keys())[method_index]].methods) - 1
            )
            
            # Randomly mutate crash option
            crash_index = random.randint(0, len(neighbor_crash_option) - 1)
            max_crash = self.tasks[list(self.tasks.keys())[crash_index]].methods[
                neighbor_method_selection[crash_index]
            ].get('max_crash_duration', 0)
            neighbor_crash_option[crash_index] = random.randint(0, max_crash)
            
            # Calculate neighbor cost
            neighbor_normal_cost, neighbor_crash_cost, neighbor_project_time, neighbor_bonus = (
                self._calculate_total_cost(neighbor_method_selection, neighbor_crash_option)
            )
            
            # Calculate net benefit
            neighbor_net_benefit = neighbor_bonus - (neighbor_normal_cost + neighbor_crash_cost)
            current_net_benefit = current_bonus - (current_normal_cost + current_crash_cost)
            
            # Decide whether to accept the new solution
            if (neighbor_net_benefit > current_net_benefit or 
                random.random() < math.exp((neighbor_net_benefit - current_net_benefit) / temperature)):
                current_method_selection = neighbor_method_selection
                current_crash_option = neighbor_crash_option
                current_normal_cost = neighbor_normal_cost
                current_crash_cost = neighbor_crash_cost
                current_project_time = neighbor_project_time
                current_bonus = neighbor_bonus
            
            # Update best solution if needed
            current_net_benefit = current_bonus - (current_normal_cost + current_crash_cost)
            if current_net_benefit > max_net_benefit:
                best_method_selection = current_method_selection
                best_crash_option = current_crash_option
                max_net_benefit = current_net_benefit
            
            # Cool down
            temperature *= self.cooling_rate
        
        # Final calculation with best solution
        best_normal_cost, best_crash_cost, best_project_time, best_bonus = (
            self._calculate_total_cost(best_method_selection, best_crash_option)
        )
        
        return {
            'method_selection': best_method_selection,
            'crash_option': best_crash_option,
            'project_time': best_project_time,
            'normal_cost': best_normal_cost,
            'crash_cost': best_crash_cost,
            'total_cost': best_normal_cost + best_crash_cost,
            'bonus': best_bonus,
            'net_benefit': best_bonus - (best_normal_cost + best_crash_cost)
        }
        

# Example usage
def create_example_project():
    """
    Create an example project with tasks having multiple methods
    """
    tasks = [
      Task('A', [
            {
                'normal_duration': 15,
                'normal_cost': 100000,
                'max_crash_duration': 3,
                'crash_cost_per_day': 2000
            },
            {
                'normal_duration': 14,
                'normal_cost': 105000,
                'max_crash_duration': 2,
                'crash_cost_per_day': 2500
            },
            {
                'normal_duration': 13,
                'normal_cost': 110000,
                'max_crash_duration': 1,
                'crash_cost_per_day': 3000
            },
            {
                'normal_duration': 12,
                'normal_cost': 115000,
                'max_crash_duration': 1,
                'crash_cost_per_day': 3500
            },
            {
                'normal_duration': 11,
                'normal_cost': 120000,
                'max_crash_duration': 0,
                'crash_cost_per_day': 0
            }
        ], []),
        Task('B', [
            {
                'normal_duration': 20,
                'normal_cost': 150000,
                'max_crash_duration': 5,
                'crash_cost_per_day': 4000
            },
            {
                'normal_duration': 19,
                'normal_cost': 158000,
                'max_crash_duration': 4,
                'crash_cost_per_day': 4500
            },
            {
                'normal_duration': 18,
                'normal_cost': 165000,
                'max_crash_duration': 3,
                'crash_cost_per_day': 5000
            },
            {
                'normal_duration': 17,
                'normal_cost': 170000,
                'max_crash_duration': 2,
                'crash_cost_per_day': 5500
            },
            {
                'normal_duration': 16,
                'normal_cost': 175000,
                'max_crash_duration': 1,
                'crash_cost_per_day': 6000
            }
        ], ['A']),
        Task('C', [
            {
                'normal_duration': 30,
                'normal_cost': 500000,
                'max_crash_duration': 7,
                'crash_cost_per_day': 10000
            },
            {
                'normal_duration': 28,
                'normal_cost': 520000,
                'max_crash_duration': 6,
                'crash_cost_per_day': 11000
            },
            {
                'normal_duration': 26,
                'normal_cost': 540000,
                'max_crash_duration': 5,
                'crash_cost_per_day': 12000
            },
            {
                'normal_duration': 25,
                'normal_cost': 555000,
                'max_crash_duration': 3,
                'crash_cost_per_day': 13000
            },
            {
                'normal_duration': 23,
                'normal_cost': 570000,
                'max_crash_duration': 2,
                'crash_cost_per_day': 14000
            }
        ], ['B']),
       Task('D', [
            {
                'normal_duration': 45,
                'normal_cost': 800000,
                'max_crash_duration': 10,
                'crash_cost_per_day': 15000
            },
            {
                'normal_duration': 42,
                'normal_cost': 840000,
                'max_crash_duration': 8,
                'crash_cost_per_day': 17000
            },
            {
                'normal_duration': 40,
                'normal_cost': 875000,
                'max_crash_duration': 6,
                'crash_cost_per_day': 18000
            },
            {
                'normal_duration': 38,
                'normal_cost': 900000,
                'max_crash_duration': 5,
                'crash_cost_per_day': 19000
            },
            {
                'normal_duration': 35,
                'normal_cost': 950000,
                'max_crash_duration': 4,
                'crash_cost_per_day': 20000
            }
        ], ['C']),
       Task('E', [
            {
                'normal_duration': 20,
                'normal_cost': 50000,
                'max_crash_duration': 5,
                'crash_cost_per_day': 1000
            },
            {
                'normal_duration': 18,
                'normal_cost': 52000,
                'max_crash_duration': 4,
                'crash_cost_per_day': 1200
            },
            {
                'normal_duration': 17,
                'normal_cost': 54000,
                'max_crash_duration': 3,
                'crash_cost_per_day': 1500
            },
            {
                'normal_duration': 16,
                'normal_cost': 56000,
                'max_crash_duration': 2,
                'crash_cost_per_day': 2000
            },
            {
                'normal_duration': 14,
                'normal_cost': 58000,
                'max_crash_duration': 1,
                'crash_cost_per_day': 2500
            }
        ], ['A']),
       Task('F', [
            {
                'normal_duration': 20,
                'normal_cost': 200000,
                'max_crash_duration': 4,
                'crash_cost_per_day': 5000
            },
            {
                'normal_duration': 19,
                'normal_cost': 210000,
                'max_crash_duration': 3,
                'crash_cost_per_day': 6000
            },
            {
                'normal_duration': 18,
                'normal_cost': 220000,
                'max_crash_duration': 2,
                'crash_cost_per_day': 7000
            },
            {
                'normal_duration': 17,
                'normal_cost': 230000,
                'max_crash_duration': 1,
                'crash_cost_per_day': 8000
            },
            {
                'normal_duration': 16,
                'normal_cost': 240000,
                'max_crash_duration': 0,
                'crash_cost_per_day': 0
            }
        ], ['E']),
       Task('G', [
            {
                'normal_duration': 15,
                'normal_cost': 100000,
                'max_crash_duration': 3,
                'crash_cost_per_day': 2500
            },
            {
                'normal_duration': 14,
                'normal_cost': 105000,
                'max_crash_duration': 2,
                'crash_cost_per_day': 3000
            },
            {
                'normal_duration': 13,
                'normal_cost': 110000,
                'max_crash_duration': 1,
                'crash_cost_per_day': 3500
            },
            {
                'normal_duration': 12,
                'normal_cost': 115000,
                'max_crash_duration': 1,
                'crash_cost_per_day': 4000
            },
            {
                'normal_duration': 11,
                'normal_cost': 120000,
                'max_crash_duration': 0,
                'crash_cost_per_day': 0
            }
        ], ['B', 'F']),
       Task('H', [
            {
                'normal_duration': 40,
                'normal_cost': 900000,
                'max_crash_duration': 8,
                'crash_cost_per_day': 10000
            },
            {
                'normal_duration': 38,
                'normal_cost': 920000,
                'max_crash_duration': 6,
                'crash_cost_per_day': 11000
            },
            {
                'normal_duration': 36,
                'normal_cost': 940000,
                'max_crash_duration': 5,
                'crash_cost_per_day': 12000
            },
            {
                'normal_duration': 34,
                'normal_cost': 960000,
                'max_crash_duration': 3,
                'crash_cost_per_day': 13000
            },
            {
                'normal_duration': 32,
                'normal_cost': 980000,
                'max_crash_duration': 2,
                'crash_cost_per_day': 14000
            }
        ], ['D', 'G']),
       Task('I', [
            {
                'normal_duration': 10,
                'normal_cost': 50000,
                'max_crash_duration': 2,
                'crash_cost_per_day': 3000
            },
            {
                'normal_duration': 9,
                'normal_cost': 53000,
                'max_crash_duration': 1,
                'crash_cost_per_day': 3500
            },
            {
                'normal_duration': 8,
                'normal_cost': 55000,
                'max_crash_duration': 1,
                'crash_cost_per_day': 4000
            },
            {
                'normal_duration': 7,
                'normal_cost': 58000,
                'max_crash_duration': 0,
                'crash_cost_per_day': 0
            },
            {
                'normal_duration': 6,
                'normal_cost': 60000,
                'max_crash_duration': 0,
                'crash_cost_per_day': 0
            }
        ], ['H']),
       Task('J', [
            {
                'normal_duration': 10,
                'normal_cost': 40000,
                'max_crash_duration': 2,
                'crash_cost_per_day': 2500
            },
            {
                'normal_duration': 9,
                'normal_cost': 42000,
                'max_crash_duration': 1,
                'crash_cost_per_day': 3000
            },
            {
                'normal_duration': 8,
                'normal_cost': 45000,
                'max_crash_duration': 1,
                'crash_cost_per_day': 3500
            },
            {
                'normal_duration': 7,
                'normal_cost': 47000,
                'max_crash_duration': 0,
                'crash_cost_per_day': 0
            },
            {
                'normal_duration': 6,
                'normal_cost': 50000,
                'max_crash_duration': 0,
                'crash_cost_per_day': 0
            }
        ], ['I'])
    ]
    
    return tasks

# Run the optimization
def main():
    # Create project tasks
    tasks = create_example_project()
    
    # Define optimization parameters
    bonus_schedule = {
        10: 34925,
        20: 84925,
        30: 159925,
        40: 259925
    }
    
    # Initialize optimizer
    optimizer = SimulatedAnnealingProjectOptimizer(
        tasks=tasks,
        max_project_duration=180,
        budget=10_000_000,
        bonus_schedule=bonus_schedule,
        initial_temperature=1000,
        cooling_rate=0.95,
        iterations=10000  # Increased iterations for more thorough search
    )
    
    # Find optimal strategy
    result = optimizer.optimize()
    
    # Print results
    print("Optimal Project Crashing Strategy:")
    print(f"Project Time: {result['project_time']} days")
    print(f"Normal Cost: ${result['normal_cost']}")
    print(f"Crash Cost: ${result['crash_cost']}")
    print(f"Total Cost: ${result['total_cost']}")
    print(f"Bonus: ${result['bonus']}")
    print(f"Net Benefit: ${result['net_benefit']}")
    print("Method Selection:", result['method_selection'])
    print("Crash Option:", result['crash_option'])

# Run the main function
if __name__ == "__main__":
    main()
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import random
import gymnasium as gym
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mission Planning Components
class MissionEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]), 
            high=np.array([1, 1, 1, 1, 1, 1]), 
            dtype=np.float32
        )
        self.reset()
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = self.observation_space.sample()
        self.steps = 0
        return self.state, {}
    
    def step(self, action):
        self.steps += 1
        self.state = np.clip(
            self.state + np.random.normal(0, 0.1, size=6),
            0,
            1
        )
        reward = self._calculate_reward(action)
        done = self.steps >= 100
        return self.state, reward, done, False, {}
    
    def _calculate_reward(self, action):
        return float(np.sum(self.state) / len(self.state))

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class MissionPlanner:
    def __init__(self):
        self.env = MissionEnvironment()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n
        ).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters())
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def train(self, num_episodes=100):
        rewards_history = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = self.q_network(state_tensor).argmax().item()
                
                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                
                if len(self.memory) > self.batch_size:
                    self._train_step()
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards_history.append(total_reward)
            
        return rewards_history
    
    def _train_step(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            max_next_q_values = self.q_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Ground Station RPA Components
class ModelManager:
    def __init__(self):
        self.models = {
            "sentiment": {
                "name": "distilbert-base-uncased-finetuned-sst-2-english",
                "type": "sentiment-analysis",
                "version": "2.0",
                "last_updated": datetime.now(),
                "performance_metrics": {
                    "accuracy": 0.92,
                    "latency_ms": []
                }
            }
        }
        
    def log_inference(self, model_name, latency_ms):
        if model_name in self.models:
            self.models[model_name]["performance_metrics"]["latency_ms"].append(latency_ms)
            
    def get_model_stats(self):
        stats = {}
        for model_name, model_info in self.models.items():
            latencies = model_info["performance_metrics"]["latency_ms"]
            stats[model_name] = {
                "avg_latency": np.mean(latencies) if latencies else 0,
                "max_latency": np.max(latencies) if latencies else 0,
                "total_inferences": len(latencies)
            }
        return stats

class CommandProcessor:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        except Exception as e:
            logger.error(f"Error initializing classifier: {e}")
            self.classifier = None

    def analyze_command(self, command_text):
        start_time = datetime.now()
        words = command_text.lower().split()
        command_type = self._detect_command_type(words)
        entities = self._extract_entities(command_text)
        
        sentiment = None
        if self.classifier:
            try:
                sentiment = self.classifier(command_text)[0]
                latency = (datetime.now() - start_time).total_seconds() * 1000
                self.model_manager.log_inference("sentiment", latency)
            except Exception as e:
                logger.error(f"Sentiment analysis error: {e}")
                sentiment = {"label": "unknown", "score": 0.0}

        return {
            'command_type': command_type,
            'words': words,
            'entities': entities,
            'sentiment': sentiment,
            'timestamp': datetime.now(),
            'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000
        }

    def _detect_command_type(self, words):
        command_types = {
            "start": ["start", "begin", "initialize", "launch"],
            "stop": ["stop", "end", "terminate", "halt"],
            "monitor": ["status", "check", "monitor", "verify"],
            "configure": ["set", "configure", "update", "modify"]
        }
        
        for cmd_type, keywords in command_types.items():
            if any(word in words for word in keywords):
                return cmd_type
        return "unknown"

    def _extract_entities(self, text):
        entities = {
            "systems": ["telemetry", "communication", "power", "thermal", "attitude"],
            "actions": ["collect", "transmit", "analyze", "store", "process"],
            "parameters": ["frequency", "bandwidth", "power", "temperature"]
        }
        
        found_entities = defaultdict(list)
        words = text.lower().split()
        
        for category, keywords in entities.items():
            found_entities[category] = [word for word in words if word in keywords]
            
        return dict(found_entities)

class WorkflowManager:
    def __init__(self):
        self.workflows = {
            "telemetry_collection": {
                "tasks": [
                    {"name": "Initialize sensors", "timeout": 30, "retry_count": 3},
                    {"name": "Start data collection", "timeout": 60, "retry_count": 2},
                    {"name": "Process telemetry", "timeout": 45, "retry_count": 1},
                    {"name": "Store data", "timeout": 30, "retry_count": 3}
                ],
                "keywords": ["telemetry", "data", "collect"]
            },
            "communication": {
                "tasks": [
                    {"name": "Check signal strength", "timeout": 15, "retry_count": 2},
                    {"name": "Establish connection", "timeout": 30, "retry_count": 3},
                    {"name": "Transmit data", "timeout": 60, "retry_count": 2},
                    {"name": "Close connection", "timeout": 15, "retry_count": 1}
                ],
                "keywords": ["communicate", "communication", "transmit", "signal"]
            },
            "system_check": {
                "tasks": [
                    {"name": "Check power systems", "timeout": 20, "retry_count": 2},
                    {"name": "Verify subsystems", "timeout": 30, "retry_count": 2},
                    {"name": "Run diagnostics", "timeout": 45, "retry_count": 1},
                    {"name": "Generate report", "timeout": 15, "retry_count": 1}
                ],
                "keywords": ["check", "verify", "status", "health"]
            }
        }
        self.execution_history = []

    def execute_workflow(self, workflow_name):
        if workflow_name not in self.workflows:
            return False, f"Workflow '{workflow_name}' not found"
            
        workflow = self.workflows[workflow_name]
        results = []
        start_time = datetime.now()
        
        for task in workflow["tasks"]:
            task_result = self._execute_task(task)
            results.append(task_result)
            
            if task_result["status"] == "Error" and task["retry_count"] > 0:
                for _ in range(task["retry_count"]):
                    if task_result["status"] != "Error":
                        break
                    task_result = self._execute_task(task)
        
        execution_record = {
            "workflow": workflow_name,
            "start_time": start_time,
            "end_time": datetime.now(),
            "results": results
        }
        
        self.execution_history.append(execution_record)
        return True, results

    def _execute_task(self, task):
        start_time = datetime.now()
        status_probs = [0.85, 0.10, 0.05]
        status = np.random.choice(['Success', 'Warning', 'Error'], p=status_probs)
        base_duration = np.random.uniform(1, 5)
        
        if "data" in task["name"].lower():
            base_duration *= 2
        
        end_time = start_time + timedelta(seconds=base_duration)
        
        return {
            'task': task["name"],
            'status': status,
            'duration': base_duration,
            'start_time': start_time,
            'end_time': end_time,
            'timeout': task["timeout"]
        }

class GroundStationRPA:
    def __init__(self):
        self.model_manager = ModelManager()
        self.command_processor = CommandProcessor(self.model_manager)
        self.workflow_manager = WorkflowManager()
        self.command_history = []
        
    def process_command(self, command_text):
        analysis = self.command_processor.analyze_command(command_text)
        
        self.command_history.append({
            'command': command_text,
            'analysis': analysis,
            'timestamp': datetime.now()
        })
        
        workflow_name = self._select_workflow(command_text, analysis)
        
        if workflow_name:
            logger.info(f"Selected workflow: {workflow_name} for command: {command_text}")
            return self.workflow_manager.execute_workflow(workflow_name)
        
        logger.warning(f"No workflow found for command: {command_text}")
        return False, "No matching workflow found"

    def _select_workflow(self, command_text, analysis):
        command_lower = command_text.lower()
        
        command_patterns = {
            "system_check": [
                "check system status",
                "system status",
                "check status",
                "verify system",
                "run diagnostics"
            ],
            "communication": [
                "initialize communication",
                "start communication",
                "begin transmission",
                "establish connection"
            ]
        }
        
        for workflow_name, patterns in command_patterns.items():
            if any(pattern in command_lower for pattern in patterns):
                return workflow_name
        
        for workflow_name, workflow in self.workflow_manager.workflows.items():
            if any(keyword in command_lower for keyword in workflow["keywords"]):
                return workflow_name
                
        return None

    def get_system_metrics(self):
        return {
            "total_commands": len(self.command_history),
            "model_stats": self.model_manager.get_model_stats(),
            "workflow_stats": self._get_workflow_stats()
        }
    
    def _get_workflow_stats(self):
        workflow_counts = defaultdict(int)
        success_rates = defaultdict(list)
        
        for execution in self.workflow_manager.execution_history:
            workflow_name = execution["workflow"]
            workflow_counts[workflow_name] += 1
            
            success_rate = len([r for r in execution["results"] if r["status"] == "Success"]) / len(execution["results"])
            success_rates[workflow_name].append(success_rate)
            
        return {
            "execution_counts": dict(workflow_counts),
            "avg_success_rates": {
                name: np.mean(rates) for name, rates in success_rates.items()
            }
        }

def generate_mission_timeline(parameters):
    """Generate a mission timeline based on parameters."""
    timeline_data = []
    start_date = datetime.now()
    
    phases = [
        {"name": "Pre-launch Preparations", "duration": 5, "category": "Preparation"},
        {"name": "Launch Operations", "duration": 1, "category": "Critical"},
        {"name": "Orbit Insertion", "duration": 2, "category": "Critical"},
        {"name": "System Checkout", "duration": 3, "category": "Verification"},
        {"name": "Primary Mission", "duration": parameters['duration'], "category": "Operations"},
        {"name": "Mission Completion", "duration": 2, "category": "Closeout"}
    ]
    
    current_date = start_date
    for phase in phases:
        end_date = current_date + timedelta(days=phase['duration'])
        timeline_data.append({
            "Task": phase['name'],
            "Start": current_date,
            "End": end_date,
            "Category": phase['category']
        })
        current_date = end_date
    
    return pd.DataFrame(timeline_data)

def main():
    st.set_page_config(page_title="Space Operations Management System", layout="wide")
    
    # Initialize session state
    if 'rpa_system' not in st.session_state:
        st.session_state.rpa_system = GroundStationRPA()
        
    # Main Navigation
    st.sidebar.title("OrbitMind")
    page = st.sidebar.radio("Select System:", ["Mission Planning", "Ground Station RPA"])
    
    if page == "Mission Planning":
        st.title("🚀 AI-Powered Mission Planning System")
        
        # Mission Configuration
        st.sidebar.header("Mission Configuration")
        mission_type = st.sidebar.selectbox(
            "Mission Type",
            ["Orbital", "Planetary", "Deep Space", "Sample Return"]
        )
        
        duration = st.sidebar.slider(
            "Mission Duration (days)",
            min_value=30,
            max_value=365,
            value=90
        )
        
        risk_tolerance = st.sidebar.slider(
            "Risk Tolerance",
            min_value=0.0,
            max_value=1.0,
            value=0.5
        )
        
        # Main content for Mission Planning
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.header("Mission Planning")
            
            if st.button("Generate Mission Plan"):
                with st.spinner("Training AI model..."):
                    planner = MissionPlanner()
                    rewards = planner.train(num_episodes=50)
                    
                    st.session_state['training_rewards'] = rewards
                    st.session_state['mission_parameters'] = {
                        'type': mission_type,
                        'duration': duration,
                        'risk_tolerance': risk_tolerance
                    }
                    
                    st.success("Mission plan generated!")
        
        with col2:
            st.header("Training Progress")
            if 'training_rewards' in st.session_state:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=st.session_state['training_rewards'],
                    mode='lines',
                    name='Training Rewards'
                ))
                fig.update_layout(
                    title='Training Progress',
                    xaxis_title='Episode',
                    yaxis_title='Total Reward'
                )
                st.plotly_chart(fig)
        
        # Mission Timeline
        st.header("Mission Timeline")
        if 'mission_parameters' in st.session_state:
            timeline_df = generate_mission_timeline(st.session_state['mission_parameters'])
            fig = px.timeline(
                timeline_df,
                x_start="Start",
                x_end="End",
                y="Task",
                color="Category",
                title="Mission Timeline"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig)
            
            # Mission Details
            st.header("Mission Details")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Duration", f"{duration} days")
            with col2:
                st.metric("Risk Level", f"{risk_tolerance:.2%}")
            with col3:
                st.metric("Mission Type", mission_type)
    
    else:  # Ground Station RPA
        st.title("🛰️ Ground Station RPA System")
        
        st.header("Command Center")
        
        with st.expander("Available Commands"):
            st.info("""
            Example commands:
            - "Check system status"
            - "Initialize communication system"
            - "Start telemetry data collection"
            - "Run system diagnostics"
            - "Begin transmission"
            - "Verify subsystems"
            """)
        
        command = st.text_input("Enter command:", key="command_input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Execute Command"):
                if command:
                    with st.spinner("Processing command..."):
                        success, results = st.session_state.rpa_system.process_command(command)
                        
                        if success:
                            st.success("Command executed successfully!")
                            
                            df = pd.DataFrame(results)
                            fig = px.timeline(
                                df,
                                x_start="start_time",
                                x_end="end_time",
                                y="task",
                                color="status",
                                title="Workflow Execution Timeline",
                                color_discrete_map={
                                    "Success": "green",
                                    "Warning": "orange",
                                    "Error": "red"
                                }
                            )
                            st.plotly_chart(fig)
                            
                            st.subheader("Task Details")
                            st.dataframe(df)
                        else:
                            st.error(f"Command execution failed: {results}")
        
        with col2:
            st.subheader("System Metrics")
            metrics = st.session_state.rpa_system.get_system_metrics()
            
            col_metrics = st.columns(3)
            with col_metrics[0]:
                st.metric("Total Commands", metrics["total_commands"])
            with col_metrics[1]:
                avg_latency = metrics["model_stats"]["sentiment"]["avg_latency"]
                st.metric("Avg. Model Latency", f"{avg_latency:.2f}ms")
            with col_metrics[2]:
                success_rates = metrics["workflow_stats"]["avg_success_rates"]
                if success_rates:
                    avg_success = np.mean(list(success_rates.values())) * 100
                    st.metric("Avg. Success Rate", f"{avg_success:.1f}%")
        
        # Command History
        st.header("Command History")
        if st.session_state.rpa_system.command_history:
            history_df = pd.DataFrame([
                {
                    'Command': entry['command'],
                    'Type': entry['analysis']['command_type'],
                    'Sentiment': entry['analysis']['sentiment']['label'],
                    'Confidence': entry['analysis']['sentiment']['score'],
                    'Processing Time (ms)': entry['analysis']['processing_time_ms'],
                    'Timestamp': entry['timestamp']
                }
                for entry in st.session_state.rpa_system.command_history
            ])
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Command Type Distribution")
                command_types = history_df['Type'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=command_types.index,
                    values=command_types.values,
                    hole=.3
                )])
                st.plotly_chart(fig)
            
            with col4:
                st.subheader("Command Processing Times")
                fig = px.box(history_df, y="Processing Time (ms)")
                st.plotly_chart(fig)
            
            st.subheader("Detailed Command History")
            st.dataframe(history_df)

if __name__ == "__main__":
    main()

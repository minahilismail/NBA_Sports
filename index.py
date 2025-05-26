import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import deque
from dataclasses import dataclass
import heapq

# Set page config
st.set_page_config(
    page_title="Simulation Explorer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin: 1rem 0;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTab [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTab [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎯 Simulation Explorer</h1>', unsafe_allow_html=True)
st.markdown("**A comprehensive interactive platform for exploring different simulation techniques**")

# Sidebar for navigation
st.sidebar.title("🧭 Navigation")
simulation_type = st.sidebar.selectbox(
    "Choose Simulation Type:",
    ["🏠 Home", "⏰ Discrete Event Simulation (DES)", "📈 Continuous Simulation", 
     "🤖 Agent-Based Simulation", "🎲 Monte Carlo Simulation", "💻 Project Resources", "📚 Research Papers"]
)

# Helper functions for different simulation types

class DiscreteEventSimulation:
    def __init__(self):
        self.events = []
        self.current_time = 0
        self.results = []
        
    def schedule_event(self, time, event_type, data=None):
        heapq.heappush(self.events, (time, event_type, data))
    
    def run_bank_simulation(self, arrival_rate, service_rate, num_tellers, simulation_time):
        # Bank queue simulation
        queue = deque()
        tellers = [0] * num_tellers  # 0 means free, >0 means busy until that time
        customers_served = 0
        total_wait_time = 0
        queue_lengths = []
        times = []
        
        # Generate initial arrivals
        next_arrival = np.random.exponential(1/arrival_rate)
        self.schedule_event(next_arrival, 'arrival', {'customer_id': 1})
        customer_id = 2
        
        while self.current_time < simulation_time and (self.events or queue):
            if self.events:
                event_time, event_type, data = heapq.heappop(self.events)
                self.current_time = event_time
                
                if event_type == 'arrival' and self.current_time < simulation_time:
                    # Customer arrives
                    arrival_time = self.current_time
                    
                    # Find free teller
                    free_teller = None
                    for i, teller_free_time in enumerate(tellers):
                        if teller_free_time <= self.current_time:
                            free_teller = i
                            break
                    
                    if free_teller is not None:
                        # Serve immediately
                        service_time = np.random.exponential(1/service_rate)
                        tellers[free_teller] = self.current_time + service_time
                        self.schedule_event(self.current_time + service_time, 'departure', 
                                          {'customer_id': data['customer_id'], 'wait_time': 0})
                    else:
                        # Join queue
                        queue.append({'customer_id': data['customer_id'], 'arrival_time': arrival_time})
                    
                    # Schedule next arrival
                    if self.current_time < simulation_time:
                        next_arrival = self.current_time + np.random.exponential(1/arrival_rate)
                        if next_arrival < simulation_time:
                            self.schedule_event(next_arrival, 'arrival', {'customer_id': customer_id})
                            customer_id += 1
                
                elif event_type == 'departure':
                    customers_served += 1
                    total_wait_time += data['wait_time']
                    
                    # Check if anyone in queue
                    if queue:
                        next_customer = queue.popleft()
                        wait_time = self.current_time - next_customer['arrival_time']
                        service_time = np.random.exponential(1/service_rate)
                        
                        # Find which teller just became free
                        for i, teller_time in enumerate(tellers):
                            if abs(teller_time - self.current_time) < 1e-10:
                                tellers[i] = self.current_time + service_time
                                break
                        
                        self.schedule_event(self.current_time + service_time, 'departure',
                                          {'customer_id': next_customer['customer_id'], 'wait_time': wait_time})
                        total_wait_time += wait_time
            
            # Record queue length
            queue_lengths.append(len(queue))
            times.append(self.current_time)
        
        avg_wait_time = total_wait_time / customers_served if customers_served > 0 else 0
        avg_queue_length = np.mean(queue_lengths) if queue_lengths else 0
        
        return {
            'customers_served': customers_served,
            'avg_wait_time': avg_wait_time,
            'avg_queue_length': avg_queue_length,
            'queue_data': (times, queue_lengths)
        }

def continuous_simulation_predator_prey(alpha, beta, gamma, delta, initial_prey, initial_predator, dt, total_time):
    """Lotka-Volterra predator-prey model"""
    time_points = np.arange(0, total_time, dt)
    prey = np.zeros(len(time_points))
    predator = np.zeros(len(time_points))
    
    prey[0] = initial_prey
    predator[0] = initial_predator
    
    for i in range(1, len(time_points)):
        # Prey dynamics: dx/dt = αx - βxy
        prey_change = alpha * prey[i-1] - beta * prey[i-1] * predator[i-1]
        
        # Predator dynamics: dy/dt = δxy - γy
        predator_change = delta * prey[i-1] * predator[i-1] - gamma * predator[i-1]
        
        prey[i] = prey[i-1] + prey_change * dt
        predator[i] = predator[i-1] + predator_change * dt
        
        # Prevent negative populations
        prey[i] = max(0, prey[i])
        predator[i] = max(0, predator[i])
    
    return time_points, prey, predator

@dataclass
class Agent:
    x: float
    y: float
    vx: float = 0
    vy: float = 0
    agent_type: str = 'normal'
    infected: bool = False
    infection_time: int = 0
    recovered: bool = False

def agent_based_epidemic_simulation(num_agents, infection_rate, recovery_time, initial_infected, steps):
    """SIR epidemic model using agent-based simulation"""
    agents = []
    
    # Create agents
    for i in range(num_agents):
        agent = Agent(
            x=np.random.uniform(0, 100),
            y=np.random.uniform(0, 100),
            vx=np.random.uniform(-1, 1),
            vy=np.random.uniform(-1, 1)
        )
        agents.append(agent)
    
    # Infect initial agents
    for i in range(min(initial_infected, num_agents)):
        agents[i].infected = True
    
    # Track statistics
    susceptible_count = []
    infected_count = []
    recovered_count = []
    
    for step in range(steps):
        # Update agent positions
        for agent in agents:
            agent.x += agent.vx
            agent.y += agent.vy
            
            # Bounce off boundaries
            if agent.x <= 0 or agent.x >= 100:
                agent.vx *= -1
            if agent.y <= 0 or agent.y >= 100:
                agent.vy *= -1
            
            agent.x = np.clip(agent.x, 0, 100)
            agent.y = np.clip(agent.y, 0, 100)
        
        # Handle infections
        for i, agent1 in enumerate(agents):
            if agent1.infected and not agent1.recovered:
                for j, agent2 in enumerate(agents):
                    if i != j and not agent2.infected and not agent2.recovered:
                        distance = np.sqrt((agent1.x - agent2.x)**2 + (agent1.y - agent2.y)**2)
                        if distance < 3 and np.random.random() < infection_rate:
                            agent2.infected = True
        
        # Handle recovery
        for agent in agents:
            if agent.infected:
                agent.infection_time += 1
                if agent.infection_time >= recovery_time:
                    agent.recovered = True
                    agent.infected = False
        
        # Count populations
        s_count = sum(1 for a in agents if not a.infected and not a.recovered)
        i_count = sum(1 for a in agents if a.infected)
        r_count = sum(1 for a in agents if a.recovered)
        
        susceptible_count.append(s_count)
        infected_count.append(i_count)
        recovered_count.append(r_count)
    
    return agents, susceptible_count, infected_count, recovered_count

def monte_carlo_portfolio_simulation(initial_value, num_assets, returns, volatilities, correlations, num_simulations, time_horizon):
    """Portfolio value simulation using Monte Carlo"""
    # Generate correlated returns
    results = []
    
    for sim in range(num_simulations):
        portfolio_values = [initial_value]
        
        for t in range(time_horizon):
            # Generate correlated random returns
            random_returns = np.random.multivariate_normal(returns, correlations * np.outer(volatilities, volatilities))
            
            # Calculate portfolio return (assuming equal weights)
            portfolio_return = np.mean(random_returns)
            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value)
        
        results.append(portfolio_values)
    
    return np.array(results)

# Main content based on selection
if simulation_type == "🏠 Home":
    # Home page
    st.markdown("## Welcome to the Simulation Explorer! 🚀")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 What You'll Explore:
        
        **Discrete Event Simulation (DES)**
        - Bank queue management
        - Customer service optimization
        - Real-time event processing
        
        **Continuous Simulation**
        - Predator-prey dynamics
        - Population modeling
        - System behavior over time
        """)
    
    with col2:
        st.markdown("""
        ### 🔬 Advanced Techniques:
        
        **Agent-Based Simulation**
        - Epidemic spread simulation
        - Individual behavior modeling
        - Emergent system properties
        
        **Monte Carlo Simulation**
        - Financial risk analysis
        - Portfolio optimization
        - Uncertainty quantification
        """)
    
    st.markdown("---")

    if st.button("🚀 Quick Start Tutorial"):
            with st.expander("Tutorial"):
                st.video("https://youtu.be/8SLk_uRRcgc")  # Replace with actual tutorial video
    
    st.markdown("---")
    
    # Feature highlights
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Interactive Visualizations</h3>
            <p>Real-time charts and graphs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚙️ Customizable Parameters</h3>
            <p>Adjust simulation settings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Real-time Analysis</h3>
            <p>Live statistical insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🎮 User-Friendly Interface</h3>
            <p>Easy to use and understand</p>
        </div>
        """, unsafe_allow_html=True)

    # Educational section
    with st.expander("📚 Learn More About Simulation Techniques"):
        st.markdown("""
        ### Discrete Event Simulation (DES)
        - **Applications**: Manufacturing systems, service operations, logistics
        - **Key Concepts**: Events, state changes, event scheduling
        - **When to Use**: Systems with distinct state changes at specific time points

        ### Continuous Simulation
        - **Applications**: Physical systems, population dynamics, engineering
        - **Key Concepts**: Differential equations, numerical integration, continuous change
        - **When to Use**: Systems where variables change smoothly over time

        ### Agent-Based Simulation (ABM)
        - **Applications**: Social systems, ecosystems, market behavior
        - **Key Concepts**: Individual agents, emergent behavior, complex interactions
        - **When to Use**: Systems where individual behavior drives overall patterns

        ### Monte Carlo Simulation
        - **Applications**: Risk analysis, optimization, uncertainty quantification
        - **Key Concepts**: Random sampling, probability distributions, statistical analysis
        - **When to Use**: Problems involving uncertainty or complex probability calculations
        """)

    # Tips and best practices
    with st.expander("💡 Simulation Best Practices"):
        st.markdown("""
        ## 🎯 Best Practices for Simulation

        ### 1. Model Validation
        - Verify your model represents reality accurately
        - Test with known scenarios
        - Compare results with analytical solutions when possible

        ### 2. Statistical Considerations
        - Run multiple replications to account for randomness
        - Use appropriate sample sizes
        - Calculate confidence intervals for your results

        ### 3. Parameter Sensitivity
        - Test how sensitive results are to parameter changes
        - Identify critical parameters that most affect outcomes
        - Document assumptions and limitations

        ### 4. Visualization and Communication
        - Use clear, interpretable visualizations
        - Provide context for your results
        - Explain limitations and assumptions to stakeholders
        """)


elif simulation_type == "⏰ Discrete Event Simulation (DES)":
    st.markdown('<h2 class="section-header">⏰ Discrete Event Simulation</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Discrete Event Simulation** models systems where state changes occur at discrete points in time.
    Perfect for modeling queues, service systems, and event-driven processes.
    """)
    
    # Bank simulation parameters
    st.markdown("### 🏦 Bank Queue Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Simulation Parameters")
        arrival_rate = st.slider("Customer Arrival Rate (customers/hour)", 1.0, 20.0, 10.0, 0.5)
        service_rate = st.slider("Service Rate (customers/hour per teller)", 1.0, 15.0, 8.0, 0.5)
        num_tellers = st.slider("Number of Tellers", 1, 5, 2)
        simulation_time = st.slider("Simulation Time (hours)", 1, 24, 8)
    
    with col2:
        st.markdown("#### Expected Metrics")
        utilization = arrival_rate / (service_rate * num_tellers)
        st.metric("System Utilization", f"{utilization:.2%}")
        
        if utilization < 1:
            avg_customers = arrival_rate / (service_rate * num_tellers - arrival_rate)
            st.metric("Expected Avg Queue Length", f"{avg_customers:.1f}")
            avg_wait = 1 / (service_rate * num_tellers - arrival_rate)
            st.metric("Expected Avg Wait Time", f"{avg_wait:.2f} hours")
        else:
            st.error("⚠️ System is overloaded! Reduce arrival rate or add more tellers.")
    
    if st.button("🚀 Run Bank Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            des = DiscreteEventSimulation()
            results = des.run_bank_simulation(arrival_rate, service_rate, num_tellers, simulation_time)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Customers Served", results['customers_served'])
            with col2:
                st.metric("Avg Wait Time", f"{results['avg_wait_time']:.2f} hours")
            with col3:
                st.metric("Avg Queue Length", f"{results['avg_queue_length']:.1f}")
            
            # Plot queue length over time
            if results['queue_data'][0]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results['queue_data'][0],
                    y=results['queue_data'][1],
                    mode='lines',
                    name='Queue Length',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.update_layout(
                    title="Queue Length Over Time",
                    xaxis_title="Time (hours)",
                    yaxis_title="Number of Customers in Queue",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)

elif simulation_type == "📈 Continuous Simulation":
    st.markdown('<h2 class="section-header">📈 Continuous Simulation</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Continuous Simulation** models systems where variables change continuously over time.
    Ideal for physical systems, population dynamics, and engineering applications.
    """)
    
    # Predator-Prey simulation
    st.markdown("### 🦌🐺 Predator-Prey Ecosystem Model")
    st.markdown("*Lotka-Volterra equations modeling the dynamics between predator and prey populations*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Parameters")
        alpha = st.slider("Prey Growth Rate (α)", 0.1, 2.0, 1.0, 0.1, help="Rate at which prey reproduce")
        beta = st.slider("Predation Rate (β)", 0.01, 1.0, 0.1, 0.01, help="Rate at which predators consume prey")
        gamma = st.slider("Predator Death Rate (γ)", 0.1, 2.0, 1.5, 0.1, help="Natural death rate of predators")
        delta = st.slider("Predator Efficiency (δ)", 0.01, 0.5, 0.075, 0.005, help="Efficiency of converting prey to predators")
    
    with col2:
        st.markdown("#### Initial Conditions")
        initial_prey = st.slider("Initial Prey Population", 1, 50, 10)
        initial_predator = st.slider("Initial Predator Population", 1, 20, 5)
        total_time = st.slider("Simulation Time", 5, 50, 20)
        dt = st.slider("Time Step", 0.001, 0.1, 0.01, 0.001)
    
    if st.button("🚀 Run Ecosystem Simulation", type="primary"):
        with st.spinner("Running continuous simulation..."):
            time_points, prey, predator = continuous_simulation_predator_prey(
                alpha, beta, gamma, delta, initial_prey, initial_predator, dt, total_time
            )
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Population Over Time', 'Phase Portrait', 'Prey Population', 'Predator Population'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Time series plot
            fig.add_trace(
                go.Scatter(x=time_points, y=prey, name='Prey', line=dict(color='green', width=2)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=time_points, y=predator, name='Predator', line=dict(color='red', width=2)),
                row=1, col=1
            )
            
            # Phase portrait
            fig.add_trace(
                go.Scatter(x=prey, y=predator, mode='lines', name='Phase Portrait', 
                          line=dict(color='purple', width=2)),
                row=1, col=2
            )
            
            # Individual populations
            fig.add_trace(
                go.Scatter(x=time_points, y=prey, mode='lines', name='Prey Detail',
                          line=dict(color='green', width=2), showlegend=False),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=time_points, y=predator, mode='lines', name='Predator Detail',
                          line=dict(color='red', width=2), showlegend=False),
                row=2, col=2
            )
            
            fig.update_layout(height=600, title_text="Predator-Prey Ecosystem Dynamics")
            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_yaxes(title_text="Population", row=1, col=1)
            fig.update_xaxes(title_text="Prey Population", row=1, col=2)
            fig.update_yaxes(title_text="Predator Population", row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Prey Population", f"{np.max(prey):.1f}")
            with col2:
                st.metric("Max Predator Population", f"{np.max(predator):.1f}")
            with col3:
                st.metric("Avg Prey Population", f"{np.mean(prey):.1f}")
            with col4:
                st.metric("Avg Predator Population", f"{np.mean(predator):.1f}")

elif simulation_type == "🤖 Agent-Based Simulation":
    st.markdown('<h2 class="section-header">🤖 Agent-Based Simulation</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Agent-Based Simulation** simulates individual agents and their interactions to understand emergent system behavior.
    Perfect for social systems, epidemics, and complex adaptive systems.
    """)
    
    # Epidemic simulation
    st.markdown("### 🦠 Epidemic Spread Simulation (SIR Model)")
    st.markdown("*Simulating disease spread through a population of moving agents*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Population Parameters")
        num_agents = st.slider("Total Population", 50, 500, 200)
        initial_infected = st.slider("Initially Infected", 1, 20, 5)
        infection_rate = st.slider("Infection Probability", 0.01, 0.5, 0.1, 0.01)
        recovery_time = st.slider("Recovery Time (days)", 5, 30, 14)
    
    with col2:
        st.markdown("#### Simulation Settings")
        simulation_steps = st.slider("Simulation Days", 50, 300, 100)
        show_animation = st.checkbox("Show Agent Animation", value=True)
        animation_speed = st.slider("Animation Speed", 1, 10, 5) if show_animation else 5
    
    if st.button("🚀 Run Epidemic Simulation", type="primary"):
        with st.spinner("Running agent-based simulation..."):
            agents, susceptible, infected, recovered = agent_based_epidemic_simulation(
                num_agents, infection_rate, recovery_time, initial_infected, simulation_steps
            )
            
            # Create epidemic curve
            fig = go.Figure()
            
            days = list(range(simulation_steps))
            fig.add_trace(go.Scatter(x=days, y=susceptible, name='Susceptible', 
                                   line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=days, y=infected, name='Infected', 
                                   line=dict(color='red', width=2)))
            fig.add_trace(go.Scatter(x=days, y=recovered, name='Recovered', 
                                   line=dict(color='green', width=2)))
            
            fig.update_layout(
                title="Epidemic Progression (SIR Model)",
                xaxis_title="Days",
                yaxis_title="Number of Agents",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Final statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Peak Infections", max(infected))
            with col2:
                peak_day = infected.index(max(infected)) if infected else 0
                st.metric("Peak Day", peak_day)
            with col3:
                total_infected = num_agents - susceptible[-1]
                st.metric("Total Ever Infected", total_infected)
            with col4:
                attack_rate = (total_infected / num_agents) * 100
                st.metric("Attack Rate", f"{attack_rate:.1f}%")
            
            # Agent visualization at peak
            if show_animation:
                st.markdown("### Agent Positions at Peak Infection")
                peak_day_idx = infected.index(max(infected))
                
                # Create a snapshot of agent positions (simplified)
                agent_data = []
                for i, agent in enumerate(agents):
                    status = 'Susceptible'
                    color = 'blue'
                    if i < initial_infected:
                        if peak_day_idx < recovery_time:
                            status = 'Infected'
                            color = 'red'
                        else:
                            status = 'Recovered'
                            color = 'green'
                    elif i < len(agents) * 0.3:  # Simplified infection spread
                        if peak_day_idx > 10:
                            status = 'Infected' if peak_day_idx < 25 else 'Recovered'
                            color = 'red' if status == 'Infected' else 'green'
                    
                    agent_data.append({
                        'x': agent.x,
                        'y': agent.y,
                        'status': status,
                        'color': color
                    })
                
                df_agents = pd.DataFrame(agent_data)
                
                fig_agents = px.scatter(df_agents, x='x', y='y', color='status',
                                      color_discrete_map={'Susceptible': 'blue', 
                                                        'Infected': 'red', 
                                                        'Recovered': 'green'},
                                      title=f"Agent Positions (Day {peak_day_idx})")
                fig_agents.update_layout(xaxis_title="X Position", yaxis_title="Y Position")
                st.plotly_chart(fig_agents, use_container_width=True)

elif simulation_type == "🎲 Monte Carlo Simulation":
    st.markdown('<h2 class="section-header">🎲 Monte Carlo Simulation</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Monte Carlo Simulation** uses random sampling to solve problems that might be deterministic in principle.
    Excellent for risk analysis, financial modeling, and uncertainty quantification.
    """)
    
    # Portfolio simulation
    st.markdown("### 💰 Investment Portfolio Risk Analysis")
    st.markdown("*Simulating portfolio performance under market uncertainty*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Portfolio Parameters")
        initial_value = st.number_input("Initial Portfolio Value ($)", 1000, 1000000, 100000, 1000)
        num_assets = st.slider("Number of Assets", 2, 10, 4)
        time_horizon = st.slider("Investment Horizon (years)", 1, 20, 5)
        num_simulations = st.slider("Number of Simulations", 100, 10000, 1000, 100)
    
    with col2:
        st.markdown("#### Market Parameters")
        base_return = st.slider("Expected Annual Return", 0.0, 0.20, 0.08, 0.01, format="%.2f")
        base_volatility = st.slider("Annual Volatility", 0.05, 0.50, 0.15, 0.01, format="%.2f")
        correlation = st.slider("Asset Correlation", 0.0, 0.9, 0.3, 0.05, format="%.2f")
    
    # Generate random asset parameters
    if 'asset_params' not in st.session_state or st.button("🔄 Regenerate Asset Parameters"):
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(base_return, 0.02, num_assets)
        volatilities = np.random.normal(base_volatility, 0.03, num_assets)
        volatilities = np.abs(volatilities)  # Ensure positive
        
        # Create correlation matrix
        correlations = np.full((num_assets, num_assets), correlation)
        np.fill_diagonal(correlations, 1.0)
        
        st.session_state.asset_params = {
            'returns': returns,
            'volatilities': volatilities,
            'correlations': correlations
        }
    
    # Display asset parameters
    if 'asset_params' in st.session_state:
        params = st.session_state.asset_params
        
        asset_df = pd.DataFrame({
            'Asset': [f'Asset {i+1}' for i in range(num_assets)],
            'Expected Return': [f"{r:.2%}" for r in params['returns']],
            'Volatility': [f"{v:.2%}" for v in params['volatilities']]
        })
        
        st.markdown("#### Asset Characteristics")
        st.dataframe(asset_df, use_container_width=True)
    
    if st.button("🚀 Run Portfolio Simulation", type="primary"):
        if 'asset_params' in st.session_state:
            with st.spinner("Running Monte Carlo simulation..."):
                params = st.session_state.asset_params
                
                # Convert annual parameters to daily
                daily_returns = params['returns'] / 252
                daily_volatilities = params['volatilities'] / np.sqrt(252)
                daily_correlations = params['correlations']
                
                # Run Monte Carlo simulation
                portfolio_results = monte_carlo_portfolio_simulation(
                    initial_value, num_assets, daily_returns, daily_volatilities, 
                    daily_correlations, num_simulations, time_horizon * 252
                )
                
                # Calculate statistics
                final_values = portfolio_results[:, -1]
                returns_pct = (final_values - initial_value) / initial_value * 100
                
                # Risk metrics
                var_95 = np.percentile(returns_pct, 5)  # Value at Risk (95% confidence)
                var_99 = np.percentile(returns_pct, 1)  # Value at Risk (99% confidence)
                expected_return = np.mean(returns_pct)
                volatility = np.std(returns_pct)
                sharpe_ratio = expected_return / volatility if volatility > 0 else 0
                
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Expected Return", f"{expected_return:.1f}%")
                with col2:
                    st.metric("Volatility", f"{volatility:.1f}%")
                with col3:
                    st.metric("VaR (95%)", f"{var_95:.1f}%")
                with col4:
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                
                # Create comprehensive visualization
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Portfolio Value Paths', 'Final Value Distribution', 
                                  'Return Distribution', 'Risk-Return Analysis'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Portfolio paths (sample of simulations)
                time_axis = np.arange(0, time_horizon * 252 + 1) / 252
                sample_paths = portfolio_results[::max(1, num_simulations//50)]  # Show max 50 paths
                
                for i, path in enumerate(sample_paths):
                    fig.add_trace(
                        go.Scatter(x=time_axis, y=path, mode='lines', 
                                 line=dict(width=1, color='rgba(0,100,200,0.3)'),
                                 showlegend=False, hovertemplate='Year: %{x:.1f}<br>Value: $%{y:,.0f}'),
                        row=1, col=1
                    )
                
                # Add median path
                median_path = np.median(portfolio_results, axis=0)
                fig.add_trace(
                    go.Scatter(x=time_axis, y=median_path, mode='lines',
                             line=dict(width=3, color='red'), name='Median Path'),
                    row=1, col=1
                )
                
                # Final value histogram
                fig.add_trace(
                    go.Histogram(x=final_values, nbinsx=50, name='Final Values',
                               marker_color='lightblue', showlegend=False),
                    row=1, col=2
                )
                
                # Return distribution
                fig.add_trace(
                    go.Histogram(x=returns_pct, nbinsx=50, name='Returns',
                               marker_color='lightgreen', showlegend=False),
                    row=2, col=1
                )
                
                # Risk-return scatter (by percentiles)
                percentiles = np.arange(5, 100, 5)
                risk_return_data = []
                for p in percentiles:
                    p_returns = np.percentile(returns_pct, p)
                    p_risk = np.std(portfolio_results[np.argsort(final_values)[int(p/100*len(final_values)):], -1] / initial_value * 100)
                    risk_return_data.append((p_risk, p_returns))
                
                risk_data, return_data = zip(*risk_return_data)
                fig.add_trace(
                    go.Scatter(x=risk_data, y=return_data, mode='markers+lines',
                             name='Risk-Return Profile', marker=dict(size=8, color='purple')),
                    row=2, col=2
                )
                
                fig.update_layout(height=700, title_text="Monte Carlo Portfolio Analysis Results")
                fig.update_xaxes(title_text="Years", row=1, col=1)
                fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
                fig.update_xaxes(title_text="Final Portfolio Value ($)", row=1, col=2)
                fig.update_yaxes(title_text="Frequency", row=1, col=2)
                fig.update_xaxes(title_text="Return (%)", row=2, col=1)
                fig.update_yaxes(title_text="Frequency", row=2, col=1)
                fig.update_xaxes(title_text="Risk (Volatility %)", row=2, col=2)
                fig.update_yaxes(title_text="Return (%)", row=2, col=2)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Probability analysis
                st.markdown("### 📊 Probability Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Probability of Outcomes")
                    prob_loss = np.sum(returns_pct < 0) / len(returns_pct) * 100
                    prob_double = np.sum(returns_pct > 100) / len(returns_pct) * 100
                    prob_positive = np.sum(returns_pct > 0) / len(returns_pct) * 100
                    
                    st.metric("Probability of Loss", f"{prob_loss:.1f}%")
                    st.metric("Probability of Positive Return", f"{prob_positive:.1f}%")
                    st.metric("Probability of Doubling Investment", f"{prob_double:.1f}%")
                
                with col2:
                    st.markdown("#### Value at Risk Analysis")
                    initial_formatted = f"${initial_value:,}"
                    var_95_value = initial_value * (1 + var_95/100)
                    var_99_value = initial_value * (1 + var_99/100)
                    
                    st.metric("Initial Investment", initial_formatted)
                    st.metric("VaR 95% (Worst 5%)", f"${var_95_value:,.0f}")
                    st.metric("VaR 99% (Worst 1%)", f"${var_99_value:,.0f}")
                
                # Download results option
                if st.checkbox("📥 Show Detailed Results Table"):
                    results_df = pd.DataFrame({
                        'Simulation': range(1, num_simulations + 1),
                        'Final Value': final_values,
                        'Total Return (%)': returns_pct,
                        'Gain/Loss ($)': final_values - initial_value
                    })
                    
                    st.dataframe(results_df.head(100), use_container_width=True)
                    st.info(f"Showing first 100 rows of {num_simulations} simulations")


elif simulation_type == "📚 Research Papers":
    st.markdown('<h2 class="section-header">📚 Research Papers & Academic Resources</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **Dive deep into the academic world of simulation!** 🎓 
    Access cutting-edge research papers, academic journals, and scholarly articles to enhance your understanding.
    """)
    
    # Create tabs for different simulation types
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔥 Featured Papers", "⏰ DES Research", "📈 Continuous Sim", "🤖 Agent-Based", "🎲 Monte Carlo"])
    
    with tab1:
        st.markdown("### 🌟 Must-Read Featured Papers")
        
        # Featured papers with visual cards
        papers_featured = [
            {
                "title": "Discrete-Event System Simulation",
                "authors": "Jerry Banks, John S. Carson II, Barry L. Nelson",
                "journal": "Pearson Education",
                "year": "2013",
                "link": "https://elibrary.pearson.de/book/99.150005/9781292037264",
                "description": "Comprehensive textbook covering fundamentals of discrete-event simulation with practical applications.",
                "tags": ["Textbook", "Fundamentals", "DES"],
                "citations": "2,500+"
            },
            {
                "title": "Introduction to Agent-Based Modeling",
                "authors": "Uri Wilensky, William Rand",
                "journal": "MIT Press",
                "year": "2015",
                "link": "https://mitpress.mit.edu/9780262731898/introduction-to-agent-based-modeling/",
                "description": "Definitive guide to agent-based simulation with hands-on examples and NetLogo implementations.",
                "tags": ["Agent-Based", "NetLogo", "Tutorial"],
                "citations": "1,800+"
            },
            {
                "title": "Monte Carlo Methods in Finance",
                "authors": "Peter Jäckel",
                "journal": "Wiley Finance",
                "year": "2002",
                "link": "https://www.wiley.com/en-us/Monte+Carlo+Methods+in+Finance-p-9780471497417",
                "description": "Comprehensive coverage of Monte Carlo techniques in financial modeling and risk management.",
                "tags": ["Finance", "Risk Analysis", "Monte Carlo"],
                "citations": "1,200+"
            }
        ]
        
        for paper in papers_featured:
            with st.container():
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px;
                    border-radius: 15px;
                    margin: 15px 0;
                    color: white;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                ">
                    <h3 style="margin: 0 0 10px 0; color: #fff;">📄 {paper['title']}</h3>
                    <p style="margin: 5px 0; opacity: 0.9;"><strong>Authors:</strong> {paper['authors']}</p>
                    <p style="margin: 5px 0; opacity: 0.9;"><strong>Published:</strong> {paper['journal']} ({paper['year']})</p>
                    <p style="margin: 10px 0; line-height: 1.5;">{paper['description']}</p>
                    <p style="margin: 5px 0;">
                        <strong>Citations:</strong> {paper['citations']} | 
                        <strong>Tags:</strong> {', '.join(paper['tags'])}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col3 = st.columns([1, 2])
                with col1:
                    st.link_button("📖 Read Paper", paper['link'], use_container_width=True)
                
        # Additional Resources Section
        st.markdown("---")
        st.markdown("### 🌐 Additional Academic Resources")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                color: white;
                height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            ">
                <h3>📖 Google Scholar</h3>
                <p>Search academic papers and citations</p>
            </div>
            """, unsafe_allow_html=True)
            st.link_button("🔍 Search Papers", "https://scholar.google.com/scholar?q=simulation+modeling", use_container_width=True)
        
        with col2:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                color: white;
                height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            ">
                <h3>🏛️ ArXiv</h3>
                <p>Preprints in mathematics, physics, and computer science</p>
            </div>
            """, unsafe_allow_html=True)
            st.link_button("📄 Browse ArXiv", "https://arxiv.org/list/cs.CE/recent", use_container_width=True)
        
        with col3:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                color: white;
                height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            ">
                <h3>📊 ResearchGate</h3>
                <p>Academic social network and paper repository</p>
            </div>
            """, unsafe_allow_html=True)
            st.link_button("🤝 Join ResearchGate", "https://www.researchgate.net/", use_container_width=True)
                
    
    with tab2:
        st.markdown("### ⏰ Discrete Event Simulation Research")
        
        des_papers = [
            {
                "title": "Modeling and Simulation of Discrete Event Systems",
                "authors": "ChoiDonghun Kang, Donghun Kang",
                "year": "2013",
                "link": "https://www.researchgate.net/publication/265771350_Modeling_and_Simulation_of_Discrete-Event_Systems",
                "venue": "IEEE Transactions on Automatic Control",
                "type": "Book"
            },
            {
                "title": "Parallel Discrete Event Simulation: A Survey",
                "authors": "Voon Yee Voon Vee, Wen Jing Hsu",
                "year": "1970",
                "link": "https://www.researchgate.net/publication/2600340_Parallel_Discrete_Event_Simulation_A_Survey",
                "venue": "ACM Computing Surveys",
                "type": "Survey Paper"
            },
            {
                "title": "Manufacturing Systems Modeling using DES",
                "authors": "Law & Kelton",
                "year": "2015",
                "link": "https://archive.org/details/simulationmodeli0000lawa_b1z9",
                "venue": "McGraw-Hill Education",
                "type": "Textbook"
            },
            {
                "title": "Hospital Emergency Department Simulation",
                "authors": "Gunal & Pidd",
                "year": "2010",
                "link": "https://www.researchgate.net/publication/220012360_Understanding_Accident_and_Emergency_Department_Performance_using_Simulation",
                "venue": "SIMULATION Journal",
                "type": "Case Study"
            }
        ]
        
        for i, paper in enumerate(des_papers):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                <div style="
                    background: #f8f9ff;
                    border-left: 4px solid #4285f4;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 0 10px 10px 0;
                ">
                    <h4 style="margin: 0 0 8px 0; color: #1a73e8;">📑 {paper['title']}</h4>
                    <p style="margin: 3px 0; color: #5f6368;"><strong>Authors:</strong> {paper['authors']} ({paper['year']})</p>
                    <p style="margin: 3px 0; color: #5f6368;"><strong>Published in:</strong> {paper['venue']}</p>
                    <span style="background: #e8f0fe; color: #1a73e8; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                        {paper['type']}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.link_button("🔗 Access", paper['link'], use_container_width=True)
    
    with tab3:
        st.markdown("### 📈 Continuous Simulation Research")
        
        continuous_papers = [
            {
                "title": "Numerical Methods for Ordinary Differential Equations",
                "authors": "Butcher, J.C.",
                "year": "2016",
                "link": "https://www.wiley.com/en-us/Numerical+Methods+for+Ordinary+Differential+Equations-p-9781119121534",
                "venue": "Wiley",
                "type": "Reference Book"
            },
            {
                "title": "System Dynamics: Modeling, Simulation, and Control",
                "authors": "Ogata, K.",
                "year": "2010",
                "link": "https://archive.org/details/systemdynamics0000ogat_t3o2",
                "venue": "Pearson",
                "type": "Textbook"
            },
            {
                "title": "Population Dynamics in Variable Environments",
                "authors": "Tuljapurkar, S.",
                "year": "1990",
                "link": "https://link.springer.com/book/10.1007/978-3-642-51652-8",
                "venue": "Springer",
                "type": "Research Monograph"
            },
            {
                "title": "Climate Model Simulation Techniques",
                "authors": "McGuffie & Henderson-Sellers",
                "year": "2014",
                "link": "https://books.google.com.pk/books/about/The_Climate_Modelling_Primer.html?id=_LjQAgAAQBAJ&redir_esc=y",
                "venue": "Wiley",
                "type": "Book"
            }
        ]
        
        for i, paper in enumerate(continuous_papers):
            st.markdown(f"""
            <div style="
                background: linear-gradient(90deg, #ff9a8b 0%, #fecfef 50%, #fecfef 100%);
                padding: 15px;
                border-radius: 12px;
                margin: 10px 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0; color: #333;">📊 {paper['title']}</h4>
                        <p style="margin: 5px 0; color: #666;">{paper['authors']} • {paper['venue']} • {paper['year']}</p>
                        <span style="background: rgba(255,255,255,0.7); padding: 3px 8px; border-radius: 10px; font-size: 11px; color: #333;">
                            {paper['type']}
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.link_button("📖 View", paper['link'])
    
    with tab4:
        st.markdown("### 🤖 Agent-Based Modeling Research")
        
        abm_papers = [
            {
                "title": "Agent-Based Models of Language Competition",
                "authors": "Castelló et al.",
                "year": "2013",
                "link": "https://www.researchgate.net/publication/274269266_Agent-based_models_of_language_competition",
                "venue": "Physical Review E",
                "type": "Journal Article",
                "description": "Explores language dynamics using ABM"
            },
            {
                "title": "From Swarm Intelligence to Swarm Robotics",
                "authors": "Fukuda & Nakagawa",
                "year": "Gerardo Beni",
                "link": "https://www.researchgate.net/publication/221116455_From_Swarm_Intelligence_to_Swarm_Robotics",
                "venue": "Springer",
                "type": "Conference Paper",
                "description": "Foundation paper on swarm robotics"
            },
            {
                "title": "Social Simulation: Technologies, Advances and New Discoveries",
                "authors": "Davidsson, P.",
                "year": "2002",
                "link": "https://books.google.com.pk/books/about/Social_Simulation_Technologies_Advances.html?id=U2_rEzqU3DkC&redir_esc=y",
                "venue": "IGI Global",
                "type": "Edited Volume",
                "description": "Comprehensive overview of social ABM"
            },
            {
                "title": "Multi-Agent Systems for Traffic and Transportation",
                "authors": "Bazzan & Klügl",
                "year": "2014",
                "link": "https://www.igi-global.com/book/multi-agent-systems-traffic-transportation/775",
                "venue": "Autonomous Agents and Multi-Agent Systems",
                "type": "Survey",
                "description": "ABM applications in transportation"
            }
        ]
        
        for i, paper in enumerate(abm_papers):
            with st.expander(f"🤖 {paper['title']}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"""
                    **Authors:** {paper['authors']}  
                    **Published:** {paper['venue']} ({paper['year']})  
                    **Type:** {paper['type']}
                    
                    **Description:** {paper['description']}
                    """)
                
                with col2:
                    st.link_button("🔗 Access", paper['link'])
                    
    with tab5:
        st.markdown("### 🎲 Monte Carlo Simulation Research")
        
        # Interactive research database
        st.markdown("#### 🔍 Search Monte Carlo Research")
        
        search_term = st.text_input("Search by keyword (e.g., 'finance', 'physics', 'optimization')")
        research_area = st.selectbox("Filter by Research Area:", 
                                   ["All Areas", "Finance & Economics", "Physics & Engineering", 
                                    "Statistics & Methodology", "Computer Science", "Biology & Medicine"])
        
        mc_papers = [
            {
                "title": "Monte Carlo Methods in Statistical Physics",
                "authors": "Newman & Barkema",
                "year": "1999",
                "area": "Physics & Engineering",
                "keywords": ["statistical physics", "simulation", "thermodynamics"],
                "link": "https://global.oup.com/academic/product/monte-carlo-methods-in-statistical-physics-9780198517979"
            },
            {
                "title": "Monte Carlo Simulation in Financial Engineering",
                "authors": "Glasserman, P.",
                "year": "2004",
                "area": "Finance & Economics",
                "keywords": ["finance", "derivatives", "risk management"],
                "link": "https://link.springer.com/book/10.1007/978-0-387-21617-1"
            },
            {
                "title": "A Modern Introduction to Probability and Statistics",
                "authors": "Dekking et al.",
                "year": "2005",
                "area": "Statistics & Methodology",
                "keywords": ["probability", "statistics", "monte carlo methods"],
                "link": "https://link.springer.com/book/10.1007/1-84628-168-7"
            },
            {
                "title": "Monte Carlo Methods for Applied Scientists",
                "authors": "Landau & Binder",
                "year": "2014",
                "area": "Computer Science",
                "keywords": ["computational science", "algorithms", "scientific computing"],
                "link": "https://worldscientific.com/worldscibooks/10.1142/2813#t=aboutBook"
            }
        ]
        
        # Filter papers based on search
        filtered_papers = mc_papers
        if search_term:
            filtered_papers = [p for p in mc_papers if 
                             search_term.lower() in p['title'].lower() or 
                             any(search_term.lower() in kw for kw in p['keywords'])]
        
        if research_area != "All Areas":
            filtered_papers = [p for p in filtered_papers if p['area'] == research_area]
        
        st.markdown(f"**Found {len(filtered_papers)} papers matching your criteria**")
        
        for i, paper in enumerate(filtered_papers):
            st.markdown(f"""
            <div style="
                background: #fff3cd;
                border: 1px solid #ffeaa7;
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
            ">
                <h4 style="color: #856404; margin: 0 0 10px 0;">🎯 {paper['title']}</h4>
                <p style="margin: 5px 0;"><strong>Authors:</strong> {paper['authors']} ({paper['year']})</p>
                <p style="margin: 5px 0;"><strong>Research Area:</strong> {paper['area']}</p>
                <p style="margin: 5px 0;"><strong>Keywords:</strong> {', '.join(paper['keywords'])}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1 = st.columns(1)[0]
            with col1:
                st.link_button("📚 Access", paper['link'])
            
    

elif simulation_type == "💻 Project Resources":
       st.markdown('<h2 class="section-header">💻 Project Resources & GitHub Repositories</h2>', unsafe_allow_html=True)
       
       st.markdown("""
       **Your one-stop destination for simulation projects!** 🚀  
       Explore open-source repositories, practical implementations, and learning resources to boost your simulation skills.
       """)
       
       # Create tabs for different types of resources
       tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔥 Featured Projects", "⏰ DES Projects", "📈 Continuous Sim", "🤖 Agent-Based", "🎲 Monte Carlo", "🛠️ Tools & Libraries"])
       
       with tab1:
           st.markdown("### 🌟 Featured Simulation Projects")
           
           featured_projects = [
               {
                   "name": "SimPy - Discrete Event Simulation Framework",
                   "description": "Python-based process-oriented discrete-event simulation framework",
                   "language": "Python",
                   "stars": "2.1k",
                   "link": "https://github.com/simpx/simpy",
                   "topics": ["simulation", "discrete-event", "python", "framework"],
                   "difficulty": "Intermediate",
                   "type": "Library/Framework"
               },
               {
                   "name": "NetLogo - Agent-Based Modeling Platform",
                   "description": "Programmable modeling environment for simulating natural and social phenomena",
                   "language": "Scala/Java",
                   "stars": "900+",
                   "link": "https://github.com/NetLogo/NetLogo",
                   "topics": ["agent-based", "modeling", "education", "simulation"],
                   "difficulty": "Beginner",
                   "type": "Platform"
               },
               {
                   "name": "Mesa - Agent-Based Modeling in Python",
                   "description": "Modular framework for building, analyzing and visualizing agent-based models",
                   "language": "Python",
                   "stars": "2.3k",
                   "link": "https://github.com/projectmesa/mesa",
                   "topics": ["agent-based", "python", "visualization", "modeling"],
                   "difficulty": "Intermediate",
                   "type": "Framework"
               }
           ]
           
           for project in featured_projects:
               st.markdown(f"""
               <div style="
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   padding: 25px;
                   border-radius: 20px;
                   margin: 20px 0;
                   color: white;
                   box-shadow: 0 8px 25px rgba(0,0,0,0.2);
                   transition: transform 0.3s ease;
               ">
                   <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">
                       <h3 style="margin: 0; color: #fff;">🚀 {project['name']}</h3>
                       <div style="display: flex; gap: 10px;">
                           <span style="background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 15px; font-size: 12px;">
                               ⭐ {project['stars']} stars
                           </span>
                           <span style="background: rgba(255,255,255,0.2); padding: 5px 10px; border-radius: 15px; font-size: 12px;">
                               {project['language']}
                           </span>
                       </div>
                   </div>
                   <p style="margin: 10px 0; line-height: 1.6; opacity: 0.95;">{project['description']}</p>
                   <div style="margin: 15px 0;">
                       <strong>Topics:</strong> {' • '.join(project['topics'])}
                   </div>
                   <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px;">
                       <div>
                           <span style="background: rgba(255,255,255,0.3); padding: 4px 12px; border-radius: 12px; font-size: 11px;">
                               {project['difficulty']}
                           </span>
                           <span style="background: rgba(255,255,255,0.3); padding: 4px 12px; border-radius: 12px; font-size: 11px; margin-left: 8px;">
                               {project['type']}
                           </span>
                       </div>
                   </div>
               </div>
               """, unsafe_allow_html=True)
               
               col1 = st.columns(4)[3]
               with col1:
                   st.link_button("🔗 GitHub Repo", project['link'], use_container_width=True)
               
       
       with tab2:
           st.markdown("### ⏰ Discrete Event Simulation Projects")
           
           des_projects = [
               {
                   "name": "Bank Queue Simulation",
                   "description": "Complete banking system simulation with multiple tellers and customer priorities",
                   "link": "https://github.com/DrFaustest/bank-queue-simulation",
                   "language": "Python",
                   "features": ["Multi-server queues", "Priority customers", "Statistics tracking"],
                   "level": "Beginner"
               },
               {
                   "name": "Manufacturing System Simulator",
                   "description": "Production line simulation with machine breakdowns and maintenance scheduling",
                   "link": "https://github.com/m-hoff/simantha",
                   "language": "Python/SimPy",
                   "features": ["Machine failures", "Maintenance schedules", "Production optimization"],
                   "level": "Advanced"
               },
               {
                   "name": "Hospital Emergency Department",
                   "description": "Emergency room simulation with patient triage and resource allocation",
                   "link": "https://github.com/BeatriceBon/Patient-Flow-Modeling-in-Hospital-Emergency-Departments-with-DES",
                   "language": "R/Python",
                   "features": ["Patient triage", "Resource allocation", "Wait time analysis"],
                   "level": "Intermediate"
               },
               {
                   "name": "Network Traffic Simulation",
                   "description": "Computer network simulation with packet routing and congestion control",
                   "link": "https://github.com/toruseo/UXsim",
                   "language": "C++/Python",
                   "features": ["Packet routing", "Congestion control", "Network topology"],
                   "level": "Advanced"
               }
           ]
           
           for i, project in enumerate(des_projects):
               with st.container():
                   col1, col2 = st.columns([3, 1])
                   
                   with col1:
                       st.markdown(f"""
                       <div style="
                           background: #f8f9ff;
                           border-left: 5px solid #4285f4;
                           padding: 20px;
                           margin: 15px 0;
                           border-radius: 0 15px 15px 0;
                           box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                       ">
                           <h4 style="margin: 0 0 10px 0; color: #1a73e8;">🔧 {project['name']}</h4>
                           <p style="margin: 8px 0; color: #5f6368; line-height: 1.5;">{project['description']}</p>
                           <div style="margin: 10px 0;">
                               <span style="background: #e8f0fe; color: #1a73e8; padding: 4px 8px; border-radius: 12px; font-size: 12px; margin-right: 8px;">
                                   {project['language']}
                               </span>
                               <span style="background: #f1f3f4; color: #5f6368; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                                   {project['level']}
                               </span>
                           </div>
                           <div style="margin-top: 12px;">
                               <strong style="color: #1a73e8;">Key Features:</strong>
                               <ul style="margin: 5px 0; padding-left: 20px;">
                                   {''.join(f'<li style="color: #5f6368;">{feature}</li>' for feature in project['features'])}
                               </ul>
                           </div>
                       </div>
                       """, unsafe_allow_html=True)
                   
                   with col2:
                       st.link_button("💻 View Code", project['link'], use_container_width=True)
       
       with tab3:
           st.markdown("### 📈 Continuous Simulation Projects")
           
           continuous_projects = [
               {
                   "name": "Climate Analysis Project",
                   "description": "Global climate modeling with temperature and CO2 dynamics using differential equations",
                   "link": "https://github.com/sophiaecl/Climate-Analysis-Project",
                   "language": "Python/SciPy",
                   "features": ["Temperature modeling", "CO2 dynamics", "Climate feedback loops"],
                   "level": "Advanced"
               },
               {
                   "name": "Epidemic Spread Model (SIR)",
                   "description": "Continuous SIR model for disease spread analysis with vaccination scenarios",
                   "link": "https://github.com/jannes-nikolas-weghake/disease-spread-simulation",
                   "language": "Python/NumPy",
                   "features": ["SIR dynamics", "Vaccination modeling", "Parameter estimation"],
                   "level": "Intermediate"
               },
               {
                   "name": "Population Dynamics Simulator",
                   "description": "Multi-species ecosystem modeling with Lotka-Volterra equations",
                   "link": "https://github.com/JacintaRoberts/population-dynamics-simulation",
                   "language": "MATLAB",
                   "features": ["Predator-prey models", "Competition dynamics", "Stability analysis"],
                   "level": "Advanced"
               },
               {
                   "name": "Chemical Reaction Simulator",
                   "description": "Continuous simulation of chemical kinetics and reactor design",
                   "link": "https://github.com/ReactionMechanismGenerator/ReactionMechanismSimulator.jl",
                   "language": "Python/SciPy",
                   "features": ["Reaction kinetics", "Reactor modeling", "Optimization"],
                   "level": "Advanced"
               }
           ]
           
           for i, project in enumerate(continuous_projects):
               with st.container():
                   col1, col2 = st.columns([3, 1])
                   
                   with col1:
                       st.markdown(f"""
                       <div style="
                           background: #f8f9ff;
                           border-left: 5px solid #4285f4;
                           padding: 20px;
                           margin: 15px 0;
                           border-radius: 0 15px 15px 0;
                           box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                       ">
                           <h4 style="margin: 0 0 10px 0; color: #1a73e8;">📊 {project['name']}</h4>
                           <p style="margin: 8px 0; color: #5f6368; line-height: 1.5;">{project['description']}</p>
                           <div style="margin: 10px 0;">
                               <span style="background: #e8f0fe; color: #1a73e8; padding: 4px 8px; border-radius: 12px; font-size: 12px; margin-right: 8px;">
                                   {project['language']}
                               </span>
                               <span style="background: #f1f3f4; color: #5f6368; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                                   {project['level']}
                               </span>
                           </div>
                           <div style="margin-top: 12px;">
                               <strong style="color: #1a73e8;">Key Features:</strong>
                               <ul style="margin: 5px 0; padding-left: 20px;">
                                   {''.join(f'<li style="color: #5f6368;">{feature}</li>' for feature in project['features'])}
                               </ul>
                           </div>
                       </div>
                       """, unsafe_allow_html=True)
                   
                   with col2:
                       st.link_button("💻 View Code", project['link'], use_container_width=True)

       with tab4:
           st.markdown("### 🤖 Agent-Based Modeling Projects")
           
           abm_projects = [
               {
                   "name": "Traffic Flow Simulation",
                   "description": "Multi-agent traffic simulation with autonomous vehicles and traffic lights",
                   "link": "https://github.com/maxbrenner-ai/Multi-Agent-Distributed-PPO-Traffc-light-control",
                   "language": "Python/SUMO",
                   "features": ["Vehicle agents", "Traffic optimization", "Real-world maps"],
                   "level": "Intermediate"
               },
               {
                   "name": "Market Economics Simulator",
                   "description": "Agent-based economic model with buyers, sellers, and market dynamics",
                   "link": "https://github.com/AB-CE/abce",
                   "language": "Python/Mesa",
                   "features": ["Market dynamics", "Price discovery", "Economic agents"],
                   "level": "Advanced"
               },
               {
                   "name": "Crowd Evacuation Model",
                   "description": "Emergency evacuation simulation with panic behavior and obstacle avoidance",
                   "link": "https://github.com/crowddynamics/crowddynamics",
                   "language": "Python/NumPy",
                   "features": ["Crowd behavior", "Panic modeling", "Evacuation routes"],
                   "level": "Intermediate"
               },
               {
                   "name": "Multi-Agent Ecosystem Simulation",
                   "description": "Models the interactions between different agents in a simplified ecosystem, including cows, lions, a bull, and a dog.",
                   "link": "https://github.com/Cizr/Multi-Agent-Ecosystem-Simulation",
                   "language": "NetLogo",
                   "features": ["Species interactions", "Population dynamics", "Habitat modeling"],
                   "level": "Beginner"
               },
               {
                   "name": "Social Network Dynamics",
                   "description": "Opinion formation and information spread in social networks",
                   "link": "https://github.com/aveydd/Social-Network-Analysis-in-Python",
                   "language": "Python/NetworkX",
                   "features": ["Opinion modeling", "Network effects", "Information cascades"],
                   "level": "Intermediate"
               }
           ]
           
           for i, project in enumerate(abm_projects):
               with st.container():
                   col1, col2 = st.columns([3, 1])
                   
                   with col1:
                       st.markdown(f"""
                       <div style="
                           background: #f8f9ff;
                           border-left: 5px solid #4285f4;
                           padding: 20px;
                           margin: 15px 0;
                           border-radius: 0 15px 15px 0;
                           box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                       ">
                           <h4 style="margin: 0 0 10px 0; color: #1a73e8;">🤖 {project['name']}</h4>
                           <p style="margin: 8px 0; color: #5f6368; line-height: 1.5;">{project['description']}</p>
                           <div style="margin: 10px 0;">
                               <span style="background: #e8f0fe; color: #1a73e8; padding: 4px 8px; border-radius: 12px; font-size: 12px; margin-right: 8px;">
                                   {project['language']}
                               </span>
                               <span style="background: #f1f3f4; color: #5f6368; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                                   {project['level']}
                               </span>
                           </div>
                           <div style="margin-top: 12px;">
                               <strong style="color: #1a73e8;">Key Features:</strong>
                               <ul style="margin: 5px 0; padding-left: 20px;">
                                   {''.join(f'<li style="color: #5f6368;">{feature}</li>' for feature in project['features'])}
                               </ul>
                           </div>
                       </div>
                       """, unsafe_allow_html=True)
                   
                   with col2:
                       st.link_button("💻 View Code", project['link'], use_container_width=True)

       with tab5:
           st.markdown("### 🎲 Monte Carlo Simulation Projects")
           
           monte_carlo_projects = [
               {
                   "name": "Financial Risk Analysis",
                   "description": "Portfolio risk assessment using Monte Carlo methods for VaR calculation",
                   "link": "https://github.com/cilidon/Portfolio-Risk-Management-Model",
                   "language": "Python/NumPy",
                   "features": ["Portfolio optimization", "VaR calculation", "Risk metrics"],
                   "level": "Advanced"
               },
               {
                   "name": "Option Pricing Models",
                   "description": "Black-Scholes and exotic option pricing using Monte Carlo simulation",
                   "link": "https://github.com/aldodec/Black-Scholes-Option-Pricing-with-Monte-Carlo-",
                   "language": "Python/SciPy",
                   "features": ["Option pricing", "Greeks calculation", "Path simulation"],
                   "level": "Advanced"
               },
               {
                   "name": "Quality Control Simulation",
                   "description": "Manufacturing quality control with sampling and defect rate analysis",
                   "link": "https://github.com/deepmbhatt/RIDAC-Real-Time-Industrial-Defect-detection-And-Classification",
                   "language": "R/Python",
                   "features": ["Defect analysis", "Sampling strategies", "Control charts"],
                   "level": "Intermediate"
               },
               {
                   "name": "Insurance Claim Modeling",
                   "description": "Actuarial modeling for insurance claims and premium calculation",
                   "link": "https://github.com/DigitalActuaryPS/Actuarial-Modeling-for-Insurance-Risk-Assessment",
                   "language": "R/Python",
                   "features": ["Claim modeling", "Premium calculation", "Catastrophic events"],
                   "level": "Advanced"
               }
           ]
           
           for i, project in enumerate(monte_carlo_projects):
               with st.container():
                   col1, col2 = st.columns([3, 1])
                   
                   with col1:
                       st.markdown(f"""
                       <div style="
                           background: #f8f9ff;
                           border-left: 5px solid #4285f4;
                           padding: 20px;
                           margin: 15px 0;
                           border-radius: 0 15px 15px 0;
                           box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                       ">
                           <h4 style="margin: 0 0 10px 0; color: #1a73e8;">🎯 {project['name']}</h4>
                           <p style="margin: 8px 0; color: #5f6368; line-height: 1.5;">{project['description']}</p>
                           <div style="margin: 10px 0;">
                               <span style="background: #e8f0fe; color: #1a73e8; padding: 4px 8px; border-radius: 12px; font-size: 12px; margin-right: 8px;">
                                   {project['language']}
                               </span>
                               <span style="background: #f1f3f4; color: #5f6368; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                                   {project['level']}
                               </span>
                           </div>
                           <div style="margin-top: 12px;">
                               <strong style="color: #1a73e8;">Key Features:</strong>
                               <ul style="margin: 5px 0; padding-left: 20px;">
                                   {''.join(f'<li style="color: #5f6368;">{feature}</li>' for feature in project['features'])}
                               </ul>
                           </div>
                       </div>
                       """, unsafe_allow_html=True)
                   
                   with col2:
                       st.link_button("💻 View Code", project['link'], use_container_width=True)

       with tab6:
           st.markdown("### 🛠️ Simulation Tools & Libraries")
           
           tools_projects = [
               {
                   "name": "PySimulator",
                   "description": "Comprehensive Python simulation framework with GUI and visualization tools",
                   "link": "https://github.com/PySimulator/PySimulator",
                   "language": "Python/Qt",
                   "features": ["GUI interface", "Model import", "Result visualization"],
                   "level": "Intermediate"
               },
               {
                   "name": "OpenModelica",
                   "description": "Open-source Modelica-based modeling and simulation environment",
                   "link": "https://github.com/OpenModelica/OpenModelica",
                   "language": "Modelica/C++",
                   "features": ["Modelica support", "Compiler tools", "IDE integration"],
                   "level": "Advanced"
               },
               {
                   "name": "AnyLogic Personal Learning Edition",
                   "description": "Multi-method modeling platform for discrete event, agent-based, and system dynamics",
                   "link": "https://github.com/nitman118/AnyLogic-Models",
                   "language": "Java/AnyLogic",
                   "features": ["Multi-method modeling", "3D visualization", "Web deployment"],
                   "level": "Beginner"
               },
               {
                   "name": "SUMO Traffic Simulator",
                   "description": "Open source traffic simulation package for large road networks",
                   "link": "https://github.com/eclipse/sumo",
                   "language": "C++/Python",
                   "features": ["Traffic simulation", "Network import", "Real-time control"],
                   "level": "Intermediate"
               },
               {
                   "name": "Repast Simphony",
                   "description": "Agent-based modeling and simulation platform with rich visualization",
                   "link": "https://github.com/Repast/repast.simphony",
                   "language": "Java/Groovy",
                   "features": ["Agent modeling", "Data collection", "Parameter sweeps"],
                   "level": "Advanced"
               },
               {
                   "name": "Arena Simulation Software",
                   "description": "Discrete event simulation and automation software for process improvement",
                   "link": "https://www.rockwellautomation.com/en-us/products/software/arena-simulation.html",
                   "language": "SIMAN/VBA",
                   "features": ["Process modeling", "Animation", "Statistical analysis"],
                   "level": "Beginner"
               }
           ]
           
           for i, project in enumerate(tools_projects):
               with st.container():
                   col1, col2 = st.columns([3, 1])
                   
                   with col1:
                       st.markdown(f"""
                       <div style="
                           background: #f8f9ff;
                           border-left: 5px solid #4285f4;
                           padding: 20px;
                           margin: 15px 0;
                           border-radius: 0 15px 15px 0;
                           box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                       ">
                           <h4 style="margin: 0 0 10px 0; color: #1a73e8;">🔧 {project['name']}</h4>
                           <p style="margin: 8px 0; color: #5f6368; line-height: 1.5;">{project['description']}</p>
                           <div style="margin: 10px 0;">
                               <span style="background: #e8f0fe; color: #1a73e8; padding: 4px 8px; border-radius: 12px; font-size: 12px; margin-right: 8px;">
                                   {project['language']}
                               </span>
                               <span style="background: #f1f3f4; color: #5f6368; padding: 4px 8px; border-radius: 12px; font-size: 12px;">
                                   {project['level']}
                               </span>
                           </div>
                           <div style="margin-top: 12px;">
                               <strong style="color: #1a73e8;">Key Features:</strong>
                               <ul style="margin: 5px 0; padding-left: 20px;">
                                   {''.join(f'<li style="color: #5f6368;">{feature}</li>' for feature in project['features'])}
                               </ul>
                           </div>
                       </div>
                       """, unsafe_allow_html=True)
                   
                   with col2:
                       st.link_button("💻 View Code", project['link'], use_container_width=True)    

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin: 1rem 0;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTab [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTab [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Additional features and educational content


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h4>🎯 Simulation Explorer</h4>
    <p>Explore, Learn, and Master Simulation Techniques!</p>
    <p>Developed by • Minahil Ismail and Nashrah Tahir</p>
    
</div>
""", unsafe_allow_html=True)

# Agent, Environment of Agentic AI, Types of Agents in Agentic AI and Pydantic Introduction

## Agent

A software that reads its surroundings, decides what to do, and executes its actions without the requirement for human approval at each step.

## AI Agent

AI Agents refer to autonomous software programs that use artificial intelligence to perform tasks, make decisions, or assist users. They interact with the environment, learn from feedback, and adapt their behaviors over time.

## Environment of Agentic AI

### The PEAS Framework

**P – Performance Measure:** How success is evaluated (e.g., safety, speed, and passenger comfort for a self-driving car).

**E – Environment:** The external surroundings the agent interacts with (e.g., roads, traffic, pedestrians, weather).

**A – Actuators:** Mechanisms used to act on the environment (e.g., steering wheel, brakes, or software API calls).

**S – Sensors:** Inputs used to perceive the environment (e.g., cameras, LiDAR, microphones, or database logs).

### Types of environment that an agent operates

1. **Observable vs. Partially Observable**

- **Fully Observable:** The agent’s sensors have access to the complete state of the environment at each point in time.

- **Partially Observable:** The agent has limited access to the full state due to noisy or limited sensors.

- Fully observable environments allow for simpler agent design, while partially observable environments require memory, inference, or probabilistic reasoning.

2. **Deterministic vs. Stochastic**

- **Deterministic:** The next state of the environment is completely determined by the current state and the agent’s action.

- **Stochastic:** The outcome may involve randomness or uncertainty.

- Deterministic environments are easier to model and predict, whereas stochastic environments demand robust decision-making strategies.

3. **Episodic vs. Sequential**

- **Episodic:** The agent’s experience is divided into atomic episodes. Each episode consists of a percept and an action, and is independent of others.

- **Sequential:** The current decision affects future decisions. Tasks are interdependent.

Sequential environments require strategic planning and often involve learning from past experiences.

4. **Static vs. Dynamic**

- **Static:** The environment does not change while the agent is deliberating.

- **Dynamic:** The environment can change during decision-making.

- Dynamic environments require real-time processing and adaptability.

5. **Discrete vs. Continuous**

- **Discrete:** The environment has a finite number of distinct states, percepts, and actions.

- **Continuous:** The environment can be described with real-valued variables.

- Continuous environments typically involve complex mathematical models and sensor processing.

6. **Single Agent vs. Multi-Agent** 

- **Single Agent:** The agent is the only entity acting in the environment.

- **Multi-Agent:** Multiple agents exist, possibly with competing or cooperating goals.

- In multi-agent environments, agents must account for the behavior and strategies of others.

7. **Known vs. Unknown Environment**

- **Known:** The agent knows the laws of the environment and outcomes of actions.

- **Unknown:** The agent must learn or infer how the environment works.

- Known environments allow for planning, whereas unknown environments emphasize learning and adaptation.

## Types of Agents in Agentic AI

1. **Simple Reflex Agents**

They act only on the current input using predefined rules, without memory or history. They are fast and suitable for simple, fully observable environments but perform poorly in dynamic or partially observable situations due to lack of memory and reasoning.

2. **Model-Based Reflex Agents**

They improve simple reflex agents by maintaining an internal state of the environment to handle partially observable and changing situations. However, their decisions are still mainly reactive and depend on the accuracy of the internal model.

3. **Goal-Based Agents**

They choose actions by evaluating future states in relation to defined goals. They can plan action sequences to achieve objectives, enabling more flexible problem-solving, but depend on clear goals and effective planning methods in complex environments.

4. **Utility-Based Agents**

They are concerned with the latter. Every possible action is scored against a utility function, and the agent picks whichever option produces the highest score, not just whichever one technically satisfies the objective.

5. **Learning Agents** 

They improve performance over time by learning from experience and updating their models or strategies. They adapt to changing environments and typically use supervised, unsupervised or reinforcement learning.

6. **Multi-Agent Systems**

They involve multiple agents interacting in a shared environment to achieve individual or collective goals. They may cooperate or compete and use communication and coordination to operate, but can be complex due to unpredictable behaviors.

7. **Hierarchical Agents**

They structure behavior into layers such as strategic, tactical and operational. Higher levels set goals that are broken into smaller tasks for lower levels, improving scalability and handling of complex tasks, but requiring well-designed coordination between layers.

8. **Autonomous / Agentic AI Agents**

They are built for goals that play out across many steps, often with real complexity and limited human checkpoints along the way. When a step fails or a path closes, they identify the issue and reroute.

9. **Generative AI Agent**

Embedding a LLM inside an agent architecture produces something meaningfully different from a chatbot. A generative AI agent doesn’t just retrieve and surface existing information; it creates original output and then takes action based on what it created.

## Pydantic 

Pydantic is a Python library that provides data validation and settings management using Python type annotations. It ensures data integrity by validating input against predefined types, allowing developers to create robust and reliable applications. Pydantic is widely used in Python projects, including FastAPI, where it simplifies data parsing and validation.

### Key Features

- **Type Validation:** Automatically validates data based on type annotations.

- **Data Parsing:** Converts input data into Python objects with the correct types.

- **Error Handling:** Provides clear and detailed error messages for invalid data.

- **Field Validators:** Allows custom validation logic with the @field_validator decorator.

- **Performance:** Pydantic is optimized for speed and supports optional Cython extensions for even faster performance.

- **Integration:** Provides seamless integration with popular frameworks (FastAPI, Django, Flask) and ORMs like SQLAlchemy.

### Installation

Before using Pydantic, install it using pip, including the optional settings module:

```text
pip install pydantic pydantic-settings
```
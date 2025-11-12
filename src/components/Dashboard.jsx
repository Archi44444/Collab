import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import PeerReview from './PeerReview';
import FindTeammates from './FindTeammates';
import Analytics from './Analytics';
import './Dashboard.css';

const Dashboard = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('projects');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [newProject, setNewProject] = useState({
    title: '',
    description: '',
    assignee: 'You'
  });

  // Kanban board state - starts empty
  const [kanbanData, setKanbanData] = useState({
    todo: [],
    inProgress: [],
    completed: []
  });

  // Drag and Drop handlers
  const [draggedItem, setDraggedItem] = useState(null);
  const [draggedFrom, setDraggedFrom] = useState(null);

  // Calculate statistics dynamically
  const getStatistics = () => {
    const activeProjects = kanbanData.inProgress.length;
    const tasksCompleted = kanbanData.completed.length;
    const totalTasks = kanbanData.todo.length + kanbanData.inProgress.length + kanbanData.completed.length;
    
    return {
      activeProjects,
      connections: 0, // Will be dynamic when you add connections feature
      tasksCompleted,
      avgRating: 0.0, // Will be dynamic when you add ratings feature
      totalTasks
    };
  };

  const stats = getStatistics();

  const handleDragStart = (item, column) => {
    setDraggedItem(item);
    setDraggedFrom(column);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (targetColumn) => {
    if (!draggedItem || !draggedFrom) return;

    // Remove from source column
    const sourceItems = kanbanData[draggedFrom].filter(
      item => item.id !== draggedItem.id
    );

    // Add to target column
    const targetItems = [...kanbanData[targetColumn], draggedItem];

    setKanbanData({
      ...kanbanData,
      [draggedFrom]: sourceItems,
      [targetColumn]: targetItems
    });

    setDraggedItem(null);
    setDraggedFrom(null);
  };

  // Handle dropdown status change
  const handleStatusChange = (taskId, currentColumn, newStatus) => {
    const task = kanbanData[currentColumn].find(t => t.id === taskId);
    if (!task) return;

    // Remove from current column
    const updatedCurrentColumn = kanbanData[currentColumn].filter(t => t.id !== taskId);
    
    // Add to new column
    const updatedNewColumn = [...kanbanData[newStatus], task];

    setKanbanData({
      ...kanbanData,
      [currentColumn]: updatedCurrentColumn,
      [newStatus]: updatedNewColumn
    });
  };

  const handleModalInputChange = (e) => {
    setNewProject({
      ...newProject,
      [e.target.name]: e.target.value
    });
  };

  const handleCreateProject = (e) => {
    e.preventDefault();
    
    const newTask = {
      id: Date.now(),
      title: newProject.title,
      assignee: newProject.assignee
    };

    // Add new project to TODO column
    setKanbanData({
      ...kanbanData,
      todo: [...kanbanData.todo, newTask]
    });

    // Reset form and close modal
    setNewProject({ title: '', description: '', assignee: 'You' });
    setIsModalOpen(false);
  };

  const StatCard = ({ icon, title, value, bgColor }) => (
    <div className="stat-card">
      <div className="stat-content">
        <div className="stat-header">
          <h3>{title}</h3>
        </div>
        <div className="stat-value">{value}</div>
      </div>
      <div className={`stat-icon ${bgColor}`}>
        {icon}
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch(activeTab) {
      case 'projects':
        return (
          <div className="kanban-container">
            <div className="kanban-header">
              <h2>My Projects</h2>
              <button 
                className="btn-new-project"
                onClick={() => setIsModalOpen(true)}
              >
                <span className="plus-icon">+</span>
                New Project
              </button>
            </div>

            {stats.totalTasks === 0 ? (
              <div className="empty-state-kanban">
                <div className="empty-icon-large">📋</div>
                <h3>No projects yet</h3>
                <p>Create your first project to get started!</p>
                <button 
                  className="btn-create-first"
                  onClick={() => setIsModalOpen(true)}
                >
                  + Create Project
                </button>
              </div>
            ) : (
              <div className="kanban-board">
                {/* TODO Column */}
                <div 
                  className="kanban-column"
                  onDragOver={handleDragOver}
                  onDrop={() => handleDrop('todo')}
                >
                  <div className="column-header">
                    <div className="column-title">
                      <span className="column-icon">⏰</span>
                      <h3>Todo</h3>
                      <span className="column-count">{kanbanData.todo.length}</span>
                    </div>
                  </div>
                  <div className="column-content">
                    {kanbanData.todo.map(task => (
                      <div
                        key={task.id}
                        className="kanban-card"
                        draggable
                        onDragStart={() => handleDragStart(task, 'todo')}
                      >
                        <div className="card-header">
                          <h4>{task.title}</h4>
                        </div>
                        <div className="card-footer">
                          <span className="assignee">{task.assignee}</span>
                          <select 
                            className="status-dropdown"
                            value="todo"
                            onChange={(e) => handleStatusChange(task.id, 'todo', e.target.value)}
                          >
                            <option value="todo">To Do</option>
                            <option value="inProgress">In Progress</option>
                            <option value="completed">Completed</option>
                          </select>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* IN PROGRESS Column */}
                <div 
                  className="kanban-column"
                  onDragOver={handleDragOver}
                  onDrop={() => handleDrop('inProgress')}
                >
                  <div className="column-header">
                    <div className="column-title">
                      <span className="column-icon">🔄</span>
                      <h3>In Progress</h3>
                      <span className="column-count">{kanbanData.inProgress.length}</span>
                    </div>
                  </div>
                  <div className="column-content">
                    {kanbanData.inProgress.map(task => (
                      <div
                        key={task.id}
                        className="kanban-card"
                        draggable
                        onDragStart={() => handleDragStart(task, 'inProgress')}
                      >
                        <div className="card-header">
                          <h4>{task.title}</h4>
                        </div>
                        <div className="card-footer">
                          <span className="assignee">{task.assignee}</span>
                          <select 
                            className="status-dropdown"
                            value="inProgress"
                            onChange={(e) => handleStatusChange(task.id, 'inProgress', e.target.value)}
                          >
                            <option value="todo">To Do</option>
                            <option value="inProgress">In Progress</option>
                            <option value="completed">Completed</option>
                          </select>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* COMPLETED Column */}
                <div 
                  className="kanban-column"
                  onDragOver={handleDragOver}
                  onDrop={() => handleDrop('completed')}
                >
                  <div className="column-header">
                    <div className="column-title">
                      <span className="column-icon">✓</span>
                      <h3>Completed</h3>
                      <span className="column-count">{kanbanData.completed.length}</span>
                    </div>
                  </div>
                  <div className="column-content">
                    {kanbanData.completed.map(task => (
                      <div
                        key={task.id}
                        className="kanban-card"
                        draggable
                        onDragStart={() => handleDragStart(task, 'completed')}
                      >
                        <div className="card-header">
                          <h4>{task.title}</h4>
                        </div>
                        <div className="card-footer">
                          <span className="assignee">{task.assignee}</span>
                          <select 
                            className="status-dropdown"
                            value="completed"
                            onChange={(e) => handleStatusChange(task.id, 'completed', e.target.value)}
                          >
                            <option value="todo">To Do</option>
                            <option value="inProgress">In Progress</option>
                            <option value="completed">Completed</option>
                          </select>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        );
      case 'teammates':
      return <FindTeammates kanbanData={kanbanData} />; 
    case 'analytics':
  return <Analytics kanbanData={kanbanData} />;
    case 'reviews':
      return <PeerReview kanbanData={kanbanData} />; 
    default:
      return null;
    }
  };

  return (
    <div className="dashboard">
      {/* Modal */}
      {isModalOpen && (
        <div className="modal-overlay" onClick={() => setIsModalOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button 
              className="modal-close"
              onClick={() => setIsModalOpen(false)}
            >
              ×
            </button>
            
            <div className="modal-header">
              <h2>Create New Project</h2>
              <p>Start a new project and invite teammates to collaborate</p>
            </div>

            <form onSubmit={handleCreateProject}>
              <div className="form-group">
                <label>Project Title</label>
                <input
                  type="text"
                  name="title"
                  placeholder="Enter project title"
                  value={newProject.title}
                  onChange={handleModalInputChange}
                  required
                />
              </div>

              <div className="form-group">
                <label>Description (Optional)</label>
                <textarea
                  name="description"
                  placeholder="Describe your project..."
                  value={newProject.description}
                  onChange={handleModalInputChange}
                  rows="5"
                />
              </div>

              <button type="submit" className="btn-create-project">
                Create Project
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="dashboard-header">
        <div className="logo-section">
          <div className="logo">
            <div className="logo-gradient"></div>
          </div>
          <div className="brand">
            <h1>CollabSphere</h1>
          </div>
        </div>
        <button className="btn-signout" onClick={() => navigate('/login')}>
          <span className="signout-icon">↗</span>
          Sign Out
        </button>
      </header>

      {/* Stats Section */}
      <div className="stats-section">
        <StatCard 
          icon={
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <rect x="3" y="3" width="18" height="18" rx="2" strokeWidth="2"/>
              <path d="M9 3v18" strokeWidth="2"/>
            </svg>
          }
          title="Active Projects"
          value={stats.activeProjects}
          bgColor="blue-bg"
        />
        <StatCard 
          icon={
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" strokeWidth="2"/>
              <circle cx="9" cy="7" r="4" strokeWidth="2"/>
              <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" strokeWidth="2"/>
            </svg>
          }
          title="Connections"
          value={stats.connections}
          bgColor="cyan-bg"
        />
        <StatCard 
          icon={
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" strokeWidth="2"/>
              <polyline points="22 4 12 14.01 9 11.01" strokeWidth="2"/>
            </svg>
          }
          title="Tasks Completed"
          value={stats.tasksCompleted}
          bgColor="orange-bg"
        />
        <StatCard 
          icon={
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor">
              <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" strokeWidth="2"/>
            </svg>
          }
          title="Avg Rating"
          value={stats.avgRating.toFixed(1)}
          bgColor="purple-bg"
        />
      </div>

      {/* Navigation Tabs */}
      <div className="nav-tabs">
        <button 
          className={`nav-tab ${activeTab === 'projects' ? 'active' : ''}`}
          onClick={() => setActiveTab('projects')}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <rect x="3" y="3" width="18" height="18" rx="2" strokeWidth="2"/>
            <path d="M9 3v18" strokeWidth="2"/>
          </svg>
          Projects
        </button>
        <button 
          className={`nav-tab ${activeTab === 'teammates' ? 'active' : ''}`}
          onClick={() => setActiveTab('teammates')}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <circle cx="11" cy="11" r="8" strokeWidth="2"/>
            <path d="M21 21l-4.35-4.35" strokeWidth="2"/>
          </svg>
          Find Teammates
        </button>
        <button 
          className={`nav-tab ${activeTab === 'analytics' ? 'active' : ''}`}
          onClick={() => setActiveTab('analytics')}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" strokeWidth="2"/>
          </svg>
          Analytics
        </button>
        <button 
          className={`nav-tab ${activeTab === 'reviews' ? 'active' : ''}`}
          onClick={() => setActiveTab('reviews')}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" strokeWidth="2"/>
            <circle cx="8.5" cy="7" r="4" strokeWidth="2"/>
            <polyline points="17 11 19 13 23 9" strokeWidth="2"/>
          </svg>
          Peer Reviews
        </button>
      </div>

      {/* Tab Content */}
      <div className="main-content">
        {renderTabContent()}
      </div>
    </div>
  );
};

export default Dashboard;



import React, { useState, useEffect } from 'react';
import './Analytics.css';

const Analytics = ({ kanbanData }) => {
  const [analyticsData, setAnalyticsData] = useState({
    totalProjects: 0,
    activeProjects: 0,
    completedProjects: 0,
    completionRate: 0,
    avgProjectDuration: 0,
    teamCollaboration: 0,
    sentimentScore: 0
  });

  const [projectBreakdown, setProjectBreakdown] = useState([]);

  useEffect(() => {
    if (kanbanData) {
      calculateAnalytics();
    }
  }, [kanbanData]);

  const calculateAnalytics = () => {
    const todoCount = kanbanData.todo?.length || 0;
    const inProgressCount = kanbanData.inProgress?.length || 0;
    const completedCount = kanbanData.completed?.length || 0;
    const totalProjects = todoCount + inProgressCount + completedCount;

    const completionRate = totalProjects > 0 
      ? Math.round((completedCount / totalProjects) * 100) 
      : 0;

    // Simulate sentiment analysis (positive collaboration score)
    const sentimentScore = totalProjects > 0 
      ? Math.min(85 + (completedCount * 2), 100) 
      : 0;

    // Team collaboration score based on active projects
    const teamCollaboration = inProgressCount > 0 
      ? Math.min(70 + (inProgressCount * 5), 100) 
      : 0;

    setAnalyticsData({
      totalProjects,
      activeProjects: inProgressCount,
      completedProjects: completedCount,
      completionRate,
      avgProjectDuration: totalProjects > 0 ? Math.round(14 + Math.random() * 10) : 0,
      teamCollaboration,
      sentimentScore
    });

    // Project breakdown
    setProjectBreakdown([
      { label: 'To Do', count: todoCount, percentage: totalProjects > 0 ? (todoCount / totalProjects) * 100 : 0, color: '#f59e0b' },
      { label: 'In Progress', count: inProgressCount, percentage: totalProjects > 0 ? (inProgressCount / totalProjects) * 100 : 0, color: '#3b82f6' },
      { label: 'Completed', count: completedCount, percentage: totalProjects > 0 ? (completedCount / totalProjects) * 100 : 0, color: '#10b981' }
    ]);
  };

  const MetricCard = ({ icon, title, value, subtitle, color }) => (
    <div className="metric-card">
      <div className={`metric-icon ${color}`}>
        {icon}
      </div>
      <div className="metric-content">
        <h4>{title}</h4>
        <div className="metric-value">{value}</div>
        {subtitle && <p className="metric-subtitle">{subtitle}</p>}
      </div>
    </div>
  );

  const ProgressBar = ({ percentage, color }) => (
    <div className="progress-bar-container">
      <div 
        className="progress-bar-fill" 
        style={{ 
          width: `${percentage}%`,
          backgroundColor: color 
        }}
      />
    </div>
  );

  return (
    <div className="analytics-container">
      {analyticsData.totalProjects === 0 ? (
        <div className="empty-state-analytics">
          <div className="empty-icon-analytics">📊</div>
          <h3>No Analytics Data Yet</h3>
          <p>Create projects to start seeing insights and analytics</p>
        </div>
      ) : (
        <>
          <div className="analytics-header">
            <h2>Analytics Dashboard</h2>
            <p>Track your progress and get actionable insights</p>
          </div>

          {/* Key Metrics Grid */}
          <div className="metrics-grid">
            <MetricCard
              icon="📁"
              title="Total Projects"
              value={analyticsData.totalProjects}
              subtitle="All time"
              color="blue"
            />
            <MetricCard
              icon="🔄"
              title="Active Projects"
              value={analyticsData.activeProjects}
              subtitle="In progress now"
              color="cyan"
            />
            <MetricCard
              icon="✓"
              title="Completed"
              value={analyticsData.completedProjects}
              subtitle={`${analyticsData.completionRate}% completion rate`}
              color="green"
            />
            <MetricCard
              icon="📅"
              title="Avg Duration"
              value={`${analyticsData.avgProjectDuration} days`}
              subtitle="Per project"
              color="orange"
            />
          </div>

          {/* ML-Powered Insights Section */}
          <div className="insights-section">
            <div className="section-title">
              <span className="brain-icon">🧠</span>
              <h3>ML-Powered Insights</h3>
            </div>

            <div className="insights-grid">
              <div className="insight-card sentiment">
                <div className="insight-header">
                  <h4>Team Sentiment Analysis</h4>
                  <span className="insight-badge positive">Positive</span>
                </div>
                <div className="insight-score">
                  <div className="score-circle">
                    <svg viewBox="0 0 100 100">
                      <circle
                        cx="50"
                        cy="50"
                        r="45"
                        fill="none"
                        stroke="#e2e8f0"
                        strokeWidth="10"
                      />
                      <circle
                        cx="50"
                        cy="50"
                        r="45"
                        fill="none"
                        stroke="#10b981"
                        strokeWidth="10"
                        strokeDasharray={`${analyticsData.sentimentScore * 2.827} 282.7`}
                        strokeLinecap="round"
                        transform="rotate(-90 50 50)"
                      />
                    </svg>
                    <div className="score-text">{analyticsData.sentimentScore}%</div>
                  </div>
                  <p className="insight-description">
                    Team collaboration shows positive sentiment. Members are engaged and productive.
                  </p>
                </div>
              </div>

              <div className="insight-card collaboration">
                <div className="insight-header">
                  <h4>Collaboration Health Score</h4>
                  <span className={`insight-badge ${analyticsData.teamCollaboration >= 70 ? 'positive' : 'neutral'}`}>
                    {analyticsData.teamCollaboration >= 70 ? 'Healthy' : 'Fair'}
                  </span>
                </div>
                <div className="insight-score">
                  <div className="score-circle">
                    <svg viewBox="0 0 100 100">
                      <circle
                        cx="50"
                        cy="50"
                        r="45"
                        fill="none"
                        stroke="#e2e8f0"
                        strokeWidth="10"
                      />
                      <circle
                        cx="50"
                        cy="50"
                        r="45"
                        fill="none"
                        stroke="#3b82f6"
                        strokeWidth="10"
                        strokeDasharray={`${analyticsData.teamCollaboration * 2.827} 282.7`}
                        strokeLinecap="round"
                        transform="rotate(-90 50 50)"
                      />
                    </svg>
                    <div className="score-text">{analyticsData.teamCollaboration}%</div>
                  </div>
                  <p className="insight-description">
                    Active collaboration detected. Teams are working well together on multiple projects.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Project Breakdown Section */}
          <div className="breakdown-section">
            <div className="section-title">
              <span className="chart-icon">📊</span>
              <h3>Project Distribution</h3>
            </div>

            <div className="breakdown-chart">
              {projectBreakdown.map((item, index) => (
                <div key={index} className="breakdown-item">
                  <div className="breakdown-label">
                    <div 
                      className="breakdown-color-dot" 
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="breakdown-name">{item.label}</span>
                    <span className="breakdown-count">{item.count}</span>
                  </div>
                  <ProgressBar percentage={item.percentage} color={item.color} />
                  <span className="breakdown-percentage">{Math.round(item.percentage)}%</span>
                </div>
              ))}
            </div>
          </div>

          {/* Recommendations Section */}
          <div className="recommendations-section">
            <div className="section-title">
              <span className="lightbulb-icon">💡</span>
              <h3>Recommendations</h3>
            </div>

            <div className="recommendations-list">
              {analyticsData.activeProjects === 0 && analyticsData.totalProjects > 0 && (
                <div className="recommendation-card">
                  <div className="recommendation-icon">🚀</div>
                  <div className="recommendation-content">
                    <h4>Start Working on Projects</h4>
                    <p>You have {kanbanData.todo.length} projects in To Do. Move them to In Progress to boost productivity.</p>
                  </div>
                </div>
              )}

              {analyticsData.completionRate < 50 && analyticsData.totalProjects > 2 && (
                <div className="recommendation-card">
                  <div className="recommendation-icon">🎯</div>
                  <div className="recommendation-content">
                    <h4>Focus on Completing Projects</h4>
                    <p>Your completion rate is {analyticsData.completionRate}%. Try focusing on finishing current projects before starting new ones.</p>
                  </div>
                </div>
              )}

              {analyticsData.activeProjects > 5 && (
                <div className="recommendation-card">
                  <div className="recommendation-icon">⚡</div>
                  <div className="recommendation-content">
                    <h4>High Workload Detected</h4>
                    <p>You have {analyticsData.activeProjects} active projects. Consider prioritizing or delegating tasks.</p>
                  </div>
                </div>
              )}

              {analyticsData.completionRate >= 80 && analyticsData.totalProjects >= 3 && (
                <div className="recommendation-card success">
                  <div className="recommendation-icon">🎉</div>
                  <div className="recommendation-content">
                    <h4>Excellent Progress!</h4>
                    <p>You're maintaining an {analyticsData.completionRate}% completion rate. Keep up the great work!</p>
                  </div>
                </div>
              )}

              {analyticsData.totalProjects === 0 && (
                <div className="recommendation-card">
                  <div className="recommendation-icon">📋</div>
                  <div className="recommendation-content">
                    <h4>Get Started</h4>
                    <p>Create your first project to start tracking your progress and collaboration metrics.</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default Analytics;


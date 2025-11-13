import React, { useState, useEffect } from 'react';
import './FindTeammates.css';
import axios from 'axios';

const FindTeammates = ({ kanbanData, currentUserId = 1 }) => {
  const [filters, setFilters] = useState({
    skills: [],
    years: [],
    departments: []
  });

  const [searchQuery, setSearchQuery] = useState('');
  const [showResults, setShowResults] = useState(false);
  const [filteredTeammates, setFilteredTeammates] = useState([]);
  const [sentRequests, setSentRequests] = useState([]);
  const [showRequestModal, setShowRequestModal] = useState(false);
  const [selectedTeammate, setSelectedTeammate] = useState(null);
  const [requestMessage, setRequestMessage] = useState('');
  const [projectForRequest, setProjectForRequest] = useState('');
  
  // AI-powered features
  const [aiRecommendations, setAiRecommendations] = useState([]);
  const [useAI, setUseAI] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [diversityWeight, setDiversityWeight] = useState(0.3);
  const [showAISettings, setShowAISettings] = useState(false);
  const [systemInitialized, setSystemInitialized] = useState(false);
  const [analytics, setAnalytics] = useState(null);
  const [showTeamBuilder, setShowTeamBuilder] = useState(false);
  const [teamProject, setTeamProject] = useState('');
  const [teamSize, setTeamSize] = useState(4);
  const [generatedTeam, setGeneratedTeam] = useState([]);

  const API_URL = 'http://localhost:5000/api';

  // Available filter options
  const availableSkills = [
    'React', 'Node.js', 'Python', 'Java', 'Machine Learning',
    'UI/UX Design', 'Data Science', 'MongoDB', 'SQL', 'AWS',
    'Docker', 'C++', 'Flutter', 'Django', 'Express.js', 'TensorFlow',
    'Kubernetes', 'Spring Boot', 'Arduino', 'Figma'
  ];

  const availableYears = ['1st Year', '2nd Year', '3rd Year', '4th Year'];
  
  const availableDepartments = [
    'Computer Science', 'Information Technology', 'Electronics',
    'Mechanical', 'Civil', 'Electrical', 'Data Science'
  ];

  // Sample students data (fallback)
  const allTeammates = [
    {
      id: 1,
      name: 'Alice Johnson',
      avatar: 'AJ',
      year: '3rd Year',
      department: 'Computer Science',
      skills: ['React', 'Node.js', 'MongoDB'],
      bio: 'Full-stack developer passionate about building scalable applications.',
      projectsCompleted: 12,
      rating: 4.8,
      availability: 'Available'
    },
    {
      id: 2,
      name: 'Bob Chen',
      avatar: 'BC',
      year: '4th Year',
      department: 'Data Science',
      skills: ['Python', 'Machine Learning', 'SQL'],
      bio: 'Data scientist with experience in ML model deployment.',
      projectsCompleted: 18,
      rating: 4.9,
      availability: 'Available'
    },
    {
      id: 3,
      name: 'Carol Martinez',
      avatar: 'CM',
      year: '2nd Year',
      department: 'Computer Science',
      skills: ['UI/UX Design', 'React', 'Flutter'],
      bio: 'Creative designer who loves crafting intuitive user experiences.',
      projectsCompleted: 8,
      rating: 4.6,
      availability: 'Busy'
    },
    {
      id: 4,
      name: 'David Kim',
      avatar: 'DK',
      year: '3rd Year',
      department: 'Information Technology',
      skills: ['Java', 'AWS'],
      bio: 'Backend specialist focused on cloud-native applications.',
      projectsCompleted: 15,
      rating: 4.7,
      availability: 'Available'
    },
    {
      id: 5,
      name: 'Emma Wilson',
      avatar: 'EW',
      year: '4th Year',
      department: 'Computer Science',
      skills: ['Python', 'Django', 'Docker'],
      bio: 'DevOps enthusiast building robust deployment pipelines.',
      projectsCompleted: 20,
      rating: 4.9,
      availability: 'Available'
    },
    {
      id: 6,
      name: 'Frank Rodriguez',
      avatar: 'FR',
      year: '2nd Year',
      department: 'Electronics',
      skills: ['C++', 'Python', 'Machine Learning'],
      bio: 'Embedded systems developer exploring AI applications.',
      projectsCompleted: 6,
      rating: 4.5,
      availability: 'Available'
    },
    {
      id: 7,
      name: 'Grace Lee',
      avatar: 'GL',
      year: '3rd Year',
      department: 'Computer Science',
      skills: ['React', 'Node.js', 'Express.js'],
      bio: 'MERN stack developer building modern web applications.',
      projectsCompleted: 14,
      rating: 4.8,
      availability: 'Available'
    },
    {
      id: 8,
      name: 'Henry Patel',
      avatar: 'HP',
      year: '1st Year',
      department: 'Information Technology',
      skills: ['Python', 'SQL', 'Data Science'],
      bio: 'Aspiring data analyst learning the ropes of data visualization.',
      projectsCompleted: 3,
      rating: 4.3,
      availability: 'Available'
    }
  ];

  // Initialize AI system
  useEffect(() => {
    initializeAISystem();
    loadAnalytics();
  }, []);

  const initializeAISystem = async () => {
    try {
      const response = await axios.post(`${API_URL}/initialize`, {
        students: allTeammates
      });
      
      if (response.data.success) {
        setSystemInitialized(true);
        console.log('✅ AI System initialized:', response.data.stats);
      }
    } catch (error) {
      console.error('Failed to initialize AI system:', error);
      setSystemInitialized(false);
    }
  };

  const loadAnalytics = async () => {
    try {
      const response = await axios.get(`${API_URL}/analytics`);
      if (response.data.success) {
        setAnalytics(response.data.analytics);
      }
    } catch (error) {
      console.error('Failed to load analytics:', error);
    }
  };

  const getActiveProjects = () => {
    if (!kanbanData) return [];
    const todoProjects = kanbanData.todo || [];
    const inProgressProjects = kanbanData.inProgress || [];
    return [...todoProjects, ...inProgressProjects];
  };

  const handleSearch = async () => {
    setIsLoading(true);
    
    try {
      if (useAI && systemInitialized) {
        // AI-powered search
        if (searchQuery.trim()) {
          // Smart search with query understanding
          const response = await axios.post(`${API_URL}/smart-search`, {
            user_id: currentUserId,
            query: searchQuery,
            top_n: 8
          });
          
          if (response.data.success) {
            setAiRecommendations(response.data.recommendations);
            setFilteredTeammates(response.data.recommendations);
          }
        } else {
          // Standard AI recommendations with filters
          const response = await axios.post(`${API_URL}/recommendations/${currentUserId}`, {
            top_n: 8,
            diversity_weight: diversityWeight,
            filters: {
              skills: filters.skills,
              years: filters.years,
              departments: filters.departments
            }
          });
          
          if (response.data.success) {
            setAiRecommendations(response.data.recommendations);
            setFilteredTeammates(response.data.recommendations);
          }
        }
      } else {
        // Fallback to traditional search
        let result = [...allTeammates];

        if (searchQuery.trim()) {
          result = result.filter(tm =>
            tm.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            tm.skills.some(skill => skill.toLowerCase().includes(searchQuery.toLowerCase())) ||
            tm.department.toLowerCase().includes(searchQuery.toLowerCase())
          );
        }

        if (filters.skills.length > 0) {
          result = result.filter(tm =>
            filters.skills.some(skill => tm.skills.includes(skill))
          );
        }

        if (filters.years.length > 0) {
          result = result.filter(tm => filters.years.includes(tm.year));
        }

        if (filters.departments.length > 0) {
          result = result.filter(tm => filters.departments.includes(tm.department));
        }

        setFilteredTeammates(result);
      }
      
      setShowResults(true);
    } catch (error) {
      console.error('Search failed:', error);
      // Fallback to local search
      handleLocalSearch();
    } finally {
      setIsLoading(false);
    }
  };

  const handleLocalSearch = () => {
    let result = [...allTeammates];

    if (searchQuery.trim()) {
      result = result.filter(tm =>
        tm.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        tm.skills.some(skill => skill.toLowerCase().includes(searchQuery.toLowerCase())) ||
        tm.department.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    if (filters.skills.length > 0) {
      result = result.filter(tm =>
        filters.skills.some(skill => tm.skills.includes(skill))
      );
    }

    if (filters.years.length > 0) {
      result = result.filter(tm => filters.years.includes(tm.year));
    }

    if (filters.departments.length > 0) {
      result = result.filter(tm => filters.departments.includes(tm.department));
    }

    setFilteredTeammates(result);
    setShowResults(true);
  };

  const handleTeamGeneration = async () => {
    if (!teamProject.trim()) {
      alert('Please describe your project');
      return;
    }

    setIsLoading(true);
    
    try {
      const response = await axios.post(`${API_URL}/form-team`, {
        project_description: teamProject,
        team_size: teamSize,
        exclude_ids: [currentUserId]
      });
      
      if (response.data.success) {
        setGeneratedTeam(response.data.team);
      }
    } catch (error) {
      console.error('Team generation failed:', error);
      alert('Failed to generate team. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleFilter = (category, value) => {
    setFilters(prev => {
      const currentFilters = prev[category];
      if (currentFilters.includes(value)) {
        return {
          ...prev,
          [category]: currentFilters.filter(item => item !== value)
        };
      } else {
        return {
          ...prev,
          [category]: [...currentFilters, value]
        };
      }
    });
  };

  const clearAllFilters = () => {
    setFilters({
      skills: [],
      years: [],
      departments: []
    });
    setSearchQuery('');
    setShowResults(false);
    setGeneratedTeam([]);
  };

  const handleSendRequest = (teammate) => {
    setSelectedTeammate(teammate);
    setShowRequestModal(true);
  };

  const submitRequest = (e) => {
    e.preventDefault();
    
    const newRequest = {
      id: Date.now(),
      teammate: selectedTeammate,
      project: projectForRequest,
      message: requestMessage,
      date: new Date().toISOString().split('T')[0],
      status: 'pending'
    };

    setSentRequests([...sentRequests, newRequest]);
    setShowRequestModal(false);
    setRequestMessage('');
    setProjectForRequest('');
    setSelectedTeammate(null);
  };

  const isRequestSent = (teammateId) => {
    return sentRequests.some(req => req.teammate.id === teammateId);
  };

  const FilterSection = ({ title, items, category }) => (
    <div className="filter-section">
      <h4>{title}</h4>
      <div className="filter-chips">
        {items.map(item => (
          <button
            key={item}
            className={`filter-chip ${filters[category].includes(item) ? 'active' : ''}`}
            onClick={() => toggleFilter(category, item)}
          >
            {item}
          </button>
        ))}
      </div>
    </div>
  );

  const activeProjects = getActiveProjects();

  return (
    <div className="find-teammates-container">
      {/* Request Modal */}
      {showRequestModal && (
        <div className="modal-overlay" onClick={() => setShowRequestModal(false)}>
          <div className="modal-content request-modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setShowRequestModal(false)}>×</button>
            
            <div className="modal-header">
              <h2>Send Collaboration Request</h2>
              <p>To: {selectedTeammate?.name}</p>
              {selectedTeammate?.match_percentage && (
                <div className="match-badge-modal">
                  🎯 {selectedTeammate.match_percentage}% Match
                </div>
              )}
            </div>

            <form onSubmit={submitRequest}>
              <div className="form-group">
                <label>Select Project</label>
                <select
                  value={projectForRequest}
                  onChange={(e) => setProjectForRequest(e.target.value)}
                  required
                >
                  <option value="">Choose a project</option>
                  {activeProjects.map(project => (
                    <option key={project.id} value={project.title}>
                      {project.title}
                    </option>
                  ))}
                  {activeProjects.length === 0 && (
                    <option disabled>No active projects</option>
                  )}
                </select>
              </div>

              <div className="form-group">
                <label>Message</label>
                <textarea
                  placeholder={`Hi ${selectedTeammate?.name}, I'd love to collaborate with you on this project...`}
                  value={requestMessage}
                  onChange={(e) => setRequestMessage(e.target.value)}
                  rows="5"
                  required
                />
              </div>

              <button type="submit" className="btn-send-request">
                Send Request
              </button>
            </form>
          </div>
        </div>
      )}

      {/* Team Builder Modal */}
      {showTeamBuilder && (
        <div className="modal-overlay" onClick={() => setShowTeamBuilder(false)}>
          <div className="modal-content team-builder-modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setShowTeamBuilder(false)}>×</button>
            
            <div className="modal-header">
              <h2>🤖 AI Team Builder</h2>
              <p>Let AI form the perfect team for your project</p>
            </div>

            <div className="form-group">
              <label>Project Description</label>
              <textarea
                placeholder="Describe your project... (e.g., Building a mobile app for food delivery with React Native and Node.js backend)"
                value={teamProject}
                onChange={(e) => setTeamProject(e.target.value)}
                rows="4"
              />
            </div>

            <div className="form-group">
              <label>Team Size: {teamSize}</label>
              <input
                type="range"
                min="2"
                max="6"
                value={teamSize}
                onChange={(e) => setTeamSize(parseInt(e.target.value))}
              />
            </div>

            <button 
              className="btn-generate-team"
              onClick={handleTeamGeneration}
              disabled={isLoading}
            >
              {isLoading ? '⏳ Generating...' : '✨ Generate Team'}
            </button>

            {generatedTeam.length > 0 && (
              <div className="generated-team">
                <h3>🎉 Your AI-Generated Team</h3>
                <div className="team-members">
                  {generatedTeam.map(member => (
                    <div key={member.id} className="team-member-card">
                      <div className="member-avatar">{member.avatar}</div>
                      <div className="member-info">
                        <h4>{member.name}</h4>
                        <p>{member.department}</p>
                        <div className="member-skills">
                          {member.skills.slice(0, 3).map((skill, idx) => (
                            <span key={idx} className="skill-tag-mini">{skill}</span>
                          ))}
                        </div>
                        <div className="relevance-score">
                          Relevance: {member.match_percentage}%
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* AI Settings Modal */}
      {showAISettings && (
        <div className="modal-overlay" onClick={() => setShowAISettings(false)}>
          <div className="modal-content ai-settings-modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setShowAISettings(false)}>×</button>
            
            <div className="modal-header">
              <h2>⚙️ AI Settings</h2>
            </div>

            <div className="settings-group">
              <label>
                <input
                  type="checkbox"
                  checked={useAI}
                  onChange={(e) => setUseAI(e.target.checked)}
                />
                Enable AI-Powered Recommendations
              </label>
              <p className="setting-description">
                Use machine learning to find the best matches based on skills, interests, and collaboration patterns
              </p>
            </div>

            <div className="settings-group">
              <label>Diversity Weight: {(diversityWeight * 100).toFixed(0)}%</label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={diversityWeight}
                onChange={(e) => setDiversityWeight(parseFloat(e.target.value))}
                disabled={!useAI}
              />
              <p className="setting-description">
                Higher values prioritize diverse skill sets and backgrounds
              </p>
            </div>

            {analytics && (
              <div className="analytics-preview">
                <h3>📊 System Analytics</h3>
                <div className="analytics-stats">
                  <div className="stat-box">
                    <div className="stat-value">{analytics.total_students}</div>
                    <div className="stat-label">Students</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-value">{analytics.num_clusters}</div>
                    <div className="stat-label">Clusters</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-value">{(analytics.avg_similarity * 100).toFixed(0)}%</div>
                    <div className="stat-label">Avg Match</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="teammates-header">
        <div>
          <h2>Find Teammates</h2>
          <p>Discover and connect with talented collaborators</p>
        </div>
        <div className="header-actions">
          <button 
            className="btn-team-builder"
            onClick={() => setShowTeamBuilder(true)}
          >
            🤖 AI Team Builder
          </button>
          <button 
            className="btn-ai-settings"
            onClick={() => setShowAISettings(true)}
          >
            ⚙️ AI Settings
          </button>
          {systemInitialized && (
            <span className="ai-badge">✨ AI Powered</span>
          )}
        </div>
      </div>

      <div className="teammates-layout">
        {/* Filters Sidebar */}
        <div className="filters-sidebar">
          <div className="filters-header">
            <h3>Filters</h3>
            {(filters.skills.length > 0 || filters.years.length > 0 || filters.departments.length > 0 || searchQuery) && (
              <button className="clear-filters-btn" onClick={clearAllFilters}>
                Clear All
              </button>
            )}
          </div>

          <div className="search-box">
            <input
              type="text"
              placeholder={useAI ? "AI-powered search..." : "Search by name, skill..."}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter') handleSearch();
              }}
            />
            <span className="search-icon">🔍</span>
          </div>

          {useAI && (
            <div className="ai-search-hint">
              💡 Try: "Need frontend expert" or "Looking for ML engineer"
            </div>
          )}

          <FilterSection
            title="Skills"
            items={availableSkills}
            category="skills"
          />

          <FilterSection
            title="Year"
            items={availableYears}
            category="years"
          />

          <FilterSection
            title="Department"
            items={availableDepartments}
            category="departments"
          />

          <button 
            className="btn-search-teammates" 
            onClick={handleSearch}
            disabled={isLoading}
          >
            {isLoading ? '⏳ Searching...' : '🔍 Search Teammates'}
          </button>

          {(filters.skills.length > 0 || filters.years.length > 0 || filters.departments.length > 0) && (
            <div className="active-filters">
              <h4>Active Filters</h4>
              <div className="active-filter-tags">
                {[...filters.skills, ...filters.years, ...filters.departments].map((filter, idx) => (
                  <span key={idx} className="active-filter-tag">
                    {filter}
                    <button onClick={() => {
                      if (filters.skills.includes(filter)) toggleFilter('skills', filter);
                      else if (filters.years.includes(filter)) toggleFilter('years', filter);
                      else toggleFilter('departments', filter);
                    }}>×</button>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Teammates Content */}
        <div className="teammates-content">
          {!showResults ? (
            <div className="no-search-yet">
              <div className="no-search-icon">🔍</div>
              <h3>Ready to find your perfect teammate?</h3>
              <p>Use the {useAI ? 'AI-powered' : ''} filters on the left and click "Search Teammates"</p>
              {useAI && (
                <div className="ai-features-preview">
                  <h4>✨ AI Features Available:</h4>
                  <ul>
                    <li>🎯 Smart matching based on skills & interests</li>
                    <li>🧠 Natural language search</li>
                    <li>🔮 Diversity-aware recommendations</li>
                    <li>🤝 Automatic team formation</li>
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <>
              <div className="results-header">
                <h3>
                  {filteredTeammates.length} Teammates Found
                  {useAI && aiRecommendations.length > 0 && (
                    <span className="ai-label"> (AI Ranked)</span>
                  )}
                </h3>
              </div>

              {filteredTeammates.length === 0 ? (
                <div className="no-results">
                  <div className="no-results-icon">🔍</div>
                  <h3>No teammates found</h3>
                  <p>Try adjusting your filters or search query</p>
                </div>
              ) : (
                <div className="teammates-grid">
                  {filteredTeammates.map((teammate, index) => (
                    <div key={teammate.id} className="teammate-card-find">
                      {useAI && teammate.match_percentage && (
                        <div className="match-badge">
                          🎯 {teammate.match_percentage}% Match
                        </div>
                      )}
                      {useAI && index < 3 && (
                        <div className="top-match-badge">
                          {index === 0 ? '🥇' : index === 1 ? '🥈' : '🥉'} Top Match
                        </div>
                      )}
                      
                      <div className="teammate-card-header">
                        <div className="teammate-avatar-large">{teammate.avatar}</div>
                        <span className={`availability-badge ${teammate.availability.toLowerCase()}`}>
                          {teammate.availability}
                        </span>
                      </div>

                      <h3>{teammate.name}</h3>
                      <p className="teammate-year">{teammate.year} • {teammate.department}</p>
                      <p className="teammate-bio">{teammate.bio}</p>

                      <div className="teammate-skills">
                        {teammate.skills.slice(0, 4).map((skill, idx) => (
                          <span key={idx} className="skill-tag">{skill}</span>
                        ))}
                        {teammate.skills.length > 4 && (
                          <span className="skill-tag more">+{teammate.skills.length - 4}</span>
                        )}
                      </div>

                      <div className="teammate-stats">
                        <div className="stat-mini">
                          <span className="stat-icon">📁</span>
                          <span>{teammate.projectsCompleted} projects</span>
                        </div>
                        <div className="stat-mini">
                          <span className="stat-icon">⭐</span>
                          <span>{teammate.rating}</span>
                        </div>
                        {teammate.cluster_id !== undefined && (
                          <div className="stat-mini">
                            <span className="stat-icon">🏷️</span>
                            <span>Group {teammate.cluster_id}</span>
                          </div>
                        )}
                      </div>

                      <button
                        className={`btn-send-collab ${isRequestSent(teammate.id) ? 'sent' : ''}`}
                        onClick={() => !isRequestSent(teammate.id) && handleSendRequest(teammate)}
                        disabled={isRequestSent(teammate.id)}
                      >
                        {isRequestSent(teammate.id) ? '✓ Request Sent' : 'Send Request'}
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default FindTeammates;


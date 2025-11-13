import React, { useState } from 'react';
import './FindTeammates.css';

const FindTeammates = ({ kanbanData }) => {
  const [filters, setFilters] = useState({
    skills: [],
    years: [],
    departments: []
  });

  const [searchQuery, setSearchQuery] = useState('');
  const [showResults, setShowResults] = useState(false); // Only show after search
  const [filteredTeammates, setFilteredTeammates] = useState([]);
  const [sentRequests, setSentRequests] = useState([]);
  const [showRequestModal, setShowRequestModal] = useState(false);
  const [selectedTeammate, setSelectedTeammate] = useState(null);
  const [requestMessage, setRequestMessage] = useState('');
  const [projectForRequest, setProjectForRequest] = useState('');

  // Available filter options
  const availableSkills = [
    'React', 'Node.js', 'Python', 'Java', 'Machine Learning',
    'UI/UX Design', 'Data Science', 'MongoDB', 'SQL', 'AWS',
    'Docker', 'C++', 'Flutter', 'Django', 'Express.js'
  ];

  const availableYears = ['1st Year', '2nd Year', '3rd Year', '4th Year'];
  
  const availableDepartments = [
    'Computer Science', 'Information Technology', 'Electronics',
    'Mechanical', 'Civil', 'Electrical', 'Data Science'
  ];

  // Get only TODO and IN PROGRESS projects from kanbanData
  const getActiveProjects = () => {
    if (!kanbanData) return [];
    
    const todoProjects = kanbanData.todo || [];
    const inProgressProjects = kanbanData.inProgress || [];
    
    return [...todoProjects, ...inProgressProjects];
  };

  // Mock teammates data - will only show after search
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

  const handleSearch = () => {
    let result = [...allTeammates];

    // Apply search query
    if (searchQuery.trim()) {
      result = result.filter(tm =>
        tm.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        tm.skills.some(skill => skill.toLowerCase().includes(searchQuery.toLowerCase())) ||
        tm.department.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Apply skill filters
    if (filters.skills.length > 0) {
      result = result.filter(tm =>
        filters.skills.some(skill => tm.skills.includes(skill))
      );
    }

    // Apply year filters
    if (filters.years.length > 0) {
      result = result.filter(tm => filters.years.includes(tm.year));
    }

    // Apply department filters
    if (filters.departments.length > 0) {
      result = result.filter(tm => filters.departments.includes(tm.department));
    }

    setFilteredTeammates(result);
    setShowResults(true);
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

      <div className="teammates-header">
        <div>
          <h2>Find Teammates</h2>
          <p>Discover and connect with talented collaborators</p>
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
              placeholder="Search by name, skill, department..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter') handleSearch();
              }}
            />
            <span className="search-icon">🔍</span>
          </div>

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

          {/* Search Button */}
          <button className="btn-search-teammates" onClick={handleSearch}>
            🔍 Search Teammates
          </button>

          {/* Active Filters Display */}
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
              <p>Use the filters on the left and click "Search Teammates" to discover collaborators</p>
            </div>
          ) : (
            <>
              <div className="results-header">
                <h3>{filteredTeammates.length} Teammates Found</h3>
              </div>

              {filteredTeammates.length === 0 ? (
                <div className="no-results">
                  <div className="no-results-icon">🔍</div>
                  <h3>No teammates found</h3>
                  <p>Try adjusting your filters or search query</p>
                </div>
              ) : (
                <div className="teammates-grid">
                  {filteredTeammates.map(teammate => (
                    <div key={teammate.id} className="teammate-card-find">
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
                        {teammate.skills.map((skill, idx) => (
                          <span key={idx} className="skill-tag">{skill}</span>
                        ))}
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


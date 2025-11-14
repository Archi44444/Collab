import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './login.css';

const Login = ({ onLogin }) => { // ✅ Accept onLogin prop
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('signIn');
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    institution: '',
    department: '',
    year: '',
    skills: [],
    profilePic: null,
    linkedinUrl: '',
    password: '',
    confirmPassword: ''
  });


  // Available skills list
  const availableSkills = [
    'React', 'Node.js', 'Python', 'Java', 'Machine Learning',
    'UI/UX Design', 'Data Science', 'MongoDB', 'SQL', 'AWS',
    'Docker', 'C++', 'Flutter', 'Django', 'Express.js',
    'TypeScript', 'GraphQL', 'Vue.js', 'Angular', 'Spring Boot'
  ];

  const handleTabClick = (tab) => setActiveTab(tab);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSkillToggle = (skill) => {
    setFormData(prev => ({
      ...prev,
      skills: prev.skills.includes(skill)
        ? prev.skills.filter(s => s !== skill)
        : [...prev.skills, skill]
    }));
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setFormData({ ...formData, profilePic: file });
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Validate skills for signup
    if (activeTab === 'signUp' && formData.skills.length === 0) {
      alert('Please select at least one skill');
      return;
    }

    // Validate password match
    if (activeTab === 'signUp' && formData.password !== formData.confirmPassword) {
      alert('Passwords do not match');
      return;
    }

    // ✅ Pass user data to App.jsx
    if (activeTab === 'signUp') {
      onLogin(formData); // Pass signup data
    } else {
      // For sign in, use demo data or fetch from backend
      onLogin({
        fullName: 'John Doe',
        email: formData.email,
        institution: 'Sample University',
        department: 'Computer Science',
        year: '3rd Year',
        skills: ['React', 'Node.js', 'Python'],
        profilePic: null,
        linkedinUrl: 'https://linkedin.com/in/johndoe'
      });
    }
    
    navigate('/Dashboard');
  };

  return (
    <div className="login-bg">
      <div className="back-home" onClick={() => navigate('/')}>
        ← Back to home
      </div>
      <div className="login-card">
        <h2>Welcome to CollabSphere</h2>
        <p>Sign in or create an account to start collaborating</p>
        <div className="login-tabs">
          <button 
            className={`login-tab${activeTab === 'signIn' ? ' active' : ''}`} 
            onClick={() => handleTabClick('signIn')}
          >
            Sign In
          </button>
          <button 
            className={`login-tab${activeTab === 'signUp' ? ' active' : ''}`} 
            onClick={() => handleTabClick('signUp')}
          >
            Sign Up
          </button>
        </div>
        
        {activeTab === 'signIn' ? (
          <form onSubmit={handleSubmit}>
            <label>Email</label>
            <input 
              name="email" 
              type="email" 
              placeholder="student@university.edu"
              value={formData.email}
              onChange={handleChange}
              required
            />
            <label>Password</label>
            <input 
              name="password" 
              type="password" 
              placeholder="Enter your password"
              value={formData.password}
              onChange={handleChange}
              required
            />
            <button className="btn-gradient" type="submit">
              Sign In
            </button>
          </form>
        ) : (
          <form onSubmit={handleSubmit} className="signup-form">
            <label>Full Name</label>
            <input 
              name="fullName" 
              type="text" 
              placeholder="John Doe"
              value={formData.fullName}
              onChange={handleChange}
              required
            />
            
            <label>Email</label>
            <input 
              name="email" 
              type="email" 
              placeholder="student@university.edu"
              value={formData.email}
              onChange={handleChange}
              required
            />
            <label>Institution Name</label>
<input 
  name="institution" 
  type="text" 
  placeholder="University of Technology"
  value={formData.institution}
  onChange={handleChange}
  required
/>

            
            <label>Department</label>
            <input 
              name="department" 
              type="text" 
              placeholder="Computer Science"
              value={formData.department}
              onChange={handleChange}
              required
            />
            
            <label>Year</label>
            <select 
              name="year" 
              value={formData.year} 
              onChange={handleChange}
              required
            >
              <option value="">Select year</option>
              <option>1st Year</option>
              <option>2nd Year</option>
              <option>3rd Year</option>
              <option>4th Year</option>
            </select>

            {/* Skills Selection - Mandatory */}
            <label className="required-label">Skills <span className="required-star">*</span></label>
            <div className="skills-container">
              {availableSkills.map(skill => (
                <button
                  key={skill}
                  type="button"
                  className={`skill-chip ${formData.skills.includes(skill) ? 'selected' : ''}`}
                  onClick={() => handleSkillToggle(skill)}
                >
                  {skill}
                  {formData.skills.includes(skill) && <span className="check-mark"> ✓</span>}
                </button>
              ))}
            </div>
            {formData.skills.length > 0 && (
              <div className="selected-skills-count">
                {formData.skills.length} skill{formData.skills.length !== 1 ? 's' : ''} selected
              </div>
            )}

            {/* Profile Picture - Optional */}
            <label className="optional-label">Profile Picture <span className="optional-tag">(Optional)</span></label>
            <div className="file-input-wrapper">
              <input 
                type="file"
                id="profilePic"
                accept="image/*"
                onChange={handleFileChange}
                className="file-input"
              />
              <label htmlFor="profilePic" className="file-label">
                {formData.profilePic ? formData.profilePic.name : 'Choose a photo'}
              </label>
            </div>

            {/* LinkedIn URL - Optional */}
            <label className="optional-label">LinkedIn Profile <span className="optional-tag">(Optional)</span></label>
            <input 
              name="linkedinUrl" 
              type="url" 
              placeholder="https://linkedin.com/in/yourprofile"
              value={formData.linkedinUrl}
              onChange={handleChange}
            />
            
            <label>Password</label>
            <input 
              name="password" 
              type="password" 
              placeholder="Create a password"
              value={formData.password}
              onChange={handleChange}
              required
            />
            
            <label>Confirm Password</label>
            <input 
              name="confirmPassword" 
              type="password" 
              placeholder="Confirm your password"
              value={formData.confirmPassword}
              onChange={handleChange}
              required
            />
            
            <button className="btn-gradient" type="submit">
              Create Account
            </button>
          </form>
        )}
      </div>
    </div>
  );
};

export default Login;



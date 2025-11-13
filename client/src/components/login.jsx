import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './login.css';

const Login = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('signIn');
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    department: '',
    year: '',
    password: '',
    confirmPassword: ''
  });

  const handleTabClick = (tab) => setActiveTab(tab);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Add your authentication logic here
    console.log('Form submitted:', formData);
    // Navigate to dashboard after successful login/signup
    navigate('/dashboard');
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
          <form onSubmit={handleSubmit}>
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


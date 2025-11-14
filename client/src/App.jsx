import React, { useState } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Home from './components/Home';
import Login from './components/login';  
import Dashboard from './components/Dashboard';
import './App.css';

function App() {
  const [userData, setUserData] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  const handleLogin = (data) => {
    setUserData(data);
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    setUserData(null);
    setIsAuthenticated(false);
  };

  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<Home />} />
        <Route 
          path="/Login" 
          element={<Login onLogin={handleLogin} />} 
        />
        <Route 
          path="/signin" 
          element={<Login onLogin={handleLogin} />} 
        />
        <Route 
          path="/Dashboard" 
          element={
            isAuthenticated && userData ? (
              <Dashboard 
                userData={userData} 
                onUpdateUserData={setUserData}
                onLogout={handleLogout}
              />
            ) : (
              <Navigate to="/Login" replace />
            )
          } 
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}

export default App;



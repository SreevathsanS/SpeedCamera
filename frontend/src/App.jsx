// src/App.jsx
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ThemeProvider } from './context/ThemeContext'; // Import ThemeProvider
import LoginPage from './pages/LoginPage';
import HomePage from './pages/HomePage';
import Navbar from './components/Navbar';

function ProtectedRoute({ children, adminOnly = false }) {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" />;
  if (adminOnly && user.role !== 'admin') return <Navigate to="/dashboard" />;
  return children;
}

function App() {
  return (
    <ThemeProvider> {/* Wrap everything in the ThemeProvider */}
      <Router>
        <AuthProvider>
          <div>
            <Navbar />
            <main style={{ padding: '2rem' }}>
              <Routes>
                <Route path="/login" element={<LoginPage />} />
                <Route path="/admin" element={<ProtectedRoute adminOnly={true}><HomePage /></ProtectedRoute>} />
                <Route path="/dashboard" element={<ProtectedRoute><HomePage /></ProtectedRoute>} />
                <Route path="*" element={<Navigate to="/login" />} />
              </Routes>
            </main>
          </div>
        </AuthProvider>
      </Router>
    </ThemeProvider>
  );
}

export default App;
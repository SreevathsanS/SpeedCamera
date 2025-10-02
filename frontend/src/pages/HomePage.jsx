// src/pages/HomePage.jsx
import React from 'react';
import { useAuth } from '../context/AuthContext';
import AdminDashboard from '../components/AdminDashboard';
import PublicDashboard from '../components/PublicDashboard';

export default function HomePage() {
  const { user } = useAuth();

  if (!user) {
    // This should ideally not happen if ProtectedRoute is working,
    // but it's good practice to handle it.
    return <p>Loading user data...</p>;
  }

  // Render the correct dashboard based on the user's role
  return user.role === 'admin' ? <AdminDashboard /> : <PublicDashboard />;
}
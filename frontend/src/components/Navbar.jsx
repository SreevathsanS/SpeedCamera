import React from 'react';
import { useAuth } from '../context/AuthContext';
import ThemeToggle from './ThemeToggle';
import styles from './Navbar.module.css';

export default function Navbar() {
  const { user, logout } = useAuth();

  return (
    <header className={styles.navbar}>
      <div className={styles.container}>
        <div className={styles.logo}>
          <span role="img" aria-label="camera emoji">📸</span> Traffic Monitor
        </div>
        <div className={styles.controls}>
          {user && (
            <>
              <span className={styles.welcome}>Welcome, {user.full_name || user.username}</span>
              <button onClick={logout} className={styles.logoutButton}>Logout</button>
            </>
          )}
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
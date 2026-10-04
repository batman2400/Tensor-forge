import React from 'react';
import { AsyncJob } from './components/AsyncJob/AsyncJob';
import { BatchPredict } from './components/BatchPredict/BatchPredict';
import { Header } from './components/Header';
import { SinglePredict } from './components/SinglePredict/SinglePredict';
import { UnlockScreen } from './components/UnlockScreen';
import { useApiKey } from './hooks/useApiKey';
import { useHealth } from './hooks/useHealth';
import { useTheme } from './hooks/useTheme';

export const App: React.FC = () => {
  const { isUnlocked, setKey, lock, authError } = useApiKey();
  const { health, isOnline } = useHealth(30000);
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="wrap">
      <Header
        health={health}
        isOnline={isOnline}
        theme={theme}
        onToggleTheme={toggleTheme}
        isUnlocked={isUnlocked}
        onLock={lock}
      />

      {!isUnlocked ? (
        <main>
          <UnlockScreen
            health={health}
            isOnline={isOnline}
            onUnlock={setKey}
            initialError={authError}
          />
        </main>
      ) : (
        <main id="main-content">
          <SinglePredict />
          <BatchPredict />
          <AsyncJob />
        </main>
      )}

      <footer className="desk-footer">
        <div>
          TensorForge 2.0 · RideEat ML Dispatch & Routing Desk
        </div>
        <div>
          {isUnlocked ? 'Desk Unlocked · Session Active' : 'Locked'}
        </div>
      </footer>
    </div>
  );
};

export default App;

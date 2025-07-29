import { useLayoutEffect, useState } from 'react';
import { Toaster } from '@/components/ui/sonner';
import './App.css';
import Pages from './pages';
import { useDataRobotInfo } from './api/user/hooks';
import i18n, { getSavedLanguage } from './i18n';

function App() {
  const { data: dataRobotInfo } = useDataRobotInfo();
  const [isReady, setIsReady] = useState(false);

  useLayoutEffect(() => {
    if (dataRobotInfo?.datarobot_account_info?.language && !getSavedLanguage()) {
      i18n.changeLanguage(dataRobotInfo.datarobot_account_info.language as string);
    }
    if (dataRobotInfo) {
      setIsReady(true);
    }
  }, [dataRobotInfo]);

  return (
    <>
      {isReady && <Pages />}
      <Toaster />
    </>
  );
}

export default App;

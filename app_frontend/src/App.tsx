/*
 * Copyright 2025 DataRobot, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { useLayoutEffect, useState } from 'react';
import { Toaster } from '@/components/ui/sonner';
import Pages from './pages';
import { useDataRobotInfo } from './api/user/hooks';
import i18n, { getSavedLanguage } from './i18n';
import { ThemeProvider } from './theme/theme-provider';

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
    <ThemeProvider>
      <div className="h-screen">
        {isReady && <Pages />}
        <Toaster />
      </div>
    </ThemeProvider>
  );
}

export default App;

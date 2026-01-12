import { Button } from '@/components/ui/button';
import addData from '@/assets/add-data.svg';
import startChatting from '@/assets/start-chatting.svg';
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogDescription,
  DialogTitle,
} from '@/components/ui/dialog';
import { useAppState } from '@/state';
import { useState } from 'react';
import { Separator } from './ui/separator';
import { useTranslation } from '@/i18n';

export const WelcomeModal = () => {
  const { showWelcome, hideWelcomeModal } = useAppState();
  const [open, setOpen] = useState(showWelcome);
  const { t } = useTranslation();

  const handleOpenChange = (modalOpen: boolean) => {
    setOpen(modalOpen);
    if (!modalOpen) {
      hideWelcomeModal();
    }
  };

  return (
    <Dialog defaultOpen={showWelcome} open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[768px]" data-testid="welcome-modal">
        <DialogHeader>
          <DialogTitle className="text-center mb-4">
            {t('Welcome to the “Talk To My Data” App')}
          </DialogTitle>
          <div className="flex justify-center gap-10">
            <div className="w-[280px]">
              <div className="grid justify-center mb-3">
                <img src={addData} alt="" />
              </div>
              <p className="text-center mn-label-large">{t('Add data')}</p>
              <DialogDescription className="text-center body-secondary">
                {t(
                  'Upload the datasets you want to analyze, no preprocessing or wrangling required!'
                )}
              </DialogDescription>
            </div>
            <div className="w-[280px]">
              <div className="grid justify-center mb-3">
                <img src={startChatting} alt="" />
              </div>
              <p className="text-center mn-label-large">{t('Start chatting')}</p>
              <DialogDescription className="text-center body-secondary">
                {t(
                  'Ask question and DataRobot automatically generates analytical code, datasets, and charts.'
                )}
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>
        <Separator className="border-t mt-6" />
        <DialogFooter>
          <Button testId="welcome-modal-hide-button" onClick={() => handleOpenChange(false)}>
            {t('Select data')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

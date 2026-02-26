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
          <DialogTitle className="mb-4 text-center">
            {t('Welcome to the “Talk To My Data” App')}
          </DialogTitle>
          <div className="flex justify-center gap-10">
            <div className="w-[280px]">
              <div className="mb-3 grid justify-center">
                <img src={addData} alt="" />
              </div>
              <p className="mn-label-large text-center">{t('Add data')}</p>
              <DialogDescription className="body-secondary text-center">
                {t(
                  'Upload the datasets you want to analyze, no preprocessing or wrangling required!'
                )}
              </DialogDescription>
            </div>
            <div className="w-[280px]">
              <div className="mb-3 grid justify-center">
                <img src={startChatting} alt="" />
              </div>
              <p className="mn-label-large text-center">{t('Start chatting')}</p>
              <DialogDescription className="body-secondary text-center">
                {t(
                  'Ask question and DataRobot automatically generates analytical code, datasets, and charts.'
                )}
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>
        <Separator className="mt-6 border-t" />
        <DialogFooter>
          <Button testId="welcome-modal-hide-button" onClick={() => handleOpenChange(false)}>
            {t('Select data')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

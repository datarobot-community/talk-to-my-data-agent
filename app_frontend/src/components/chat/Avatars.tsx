import { MessageCircle } from 'lucide-react';
import DataRobotLogo from '@/assets/DataRobotLogo_black.svg';
import { Link } from 'react-router-dom';
import { useTranslation } from '@/i18n';
import { ROUTES } from '@/pages/routes';

export const DataRobotAvatar = () => {
  const { t } = useTranslation();

  return (
    <div className="body text-center text-primary-foreground">
      {/* The logo carries no alt: the message header already renders "DataRobot" as
          visible text beside it, so the link is named for where it goes instead. */}
      <Link to={ROUTES.DATA} aria-label={t('Go to data')}>
        <img src={DataRobotLogo} alt="" />
      </Link>
    </div>
  );
};

export const UserAvatar = () => (
  <div className="inline-flex size-6 flex-col items-center justify-center gap-2.5 overflow-hidden rounded-[100px] bg-[#7c97f8] p-2.5">
    <div className="body text-center text-primary-foreground">
      <MessageCircle className="size-4" />
    </div>
  </div>
);

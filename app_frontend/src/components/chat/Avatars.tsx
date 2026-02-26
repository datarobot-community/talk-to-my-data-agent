import { MessageCircle } from 'lucide-react';
import DataRobotLogo from '@/assets/DataRobotLogo_black.svg';
import { useNavigate } from 'react-router-dom';
import { ROUTES } from '@/pages/routes';

export const DataRobotAvatar = () => {
  const navigate = useNavigate();

  return (
    <div className="body text-center text-primary-foreground">
      <img
        src={DataRobotLogo}
        alt=""
        className="cursor-pointer"
        onClick={() => navigate(ROUTES.DATA)}
      />
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

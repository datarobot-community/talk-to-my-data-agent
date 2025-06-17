import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faComment } from "@fortawesome/free-regular-svg-icons/faComment";
import DataRobotLogo from "@/assets/DataRobotLogo_black.svg";
import { useNavigate } from "react-router-dom";
import { ROUTES } from "@/pages/routes";

export const DataRobotAvatar = () => {
  const navigate = useNavigate();
  
  return (
    <div className="text-center text-sm font-normal leading-tight text-primary-foreground">
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
  <div className="w-6 h-6 p-2.5 bg-primary rounded-[100px] flex-col justify-center items-center gap-2.5 inline-flex overflow-hidden">
    <div className="text-center text-sm font-normal leading-tight text-primary-foreground">
      <FontAwesomeIcon icon={faComment} />
    </div>
  </div>
);

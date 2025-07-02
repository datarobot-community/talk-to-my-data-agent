import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faTable } from "@fortawesome/free-solid-svg-icons/faTable";
import { faComments } from "@fortawesome/free-regular-svg-icons/faComments";
import drLogo from "@/assets/DataRobot_white.svg";
import {
  SidebarMenu,
  SidebarMenuOptionType,
} from "@/components/ui-custom/sidebar-menu";
import { WelcomeModal } from "./WelcomeModal";
import { AddDataModal } from "./AddDataModal";
import { ROUTES, generateChatRoute } from "@/pages/routes";
import { Separator } from "@radix-ui/react-separator";
import { NewChatModal } from "./NewChatModal";
import loader from "@/assets/loader.svg";
import { useGeneratedDictionaries } from "@/api/dictionaries/hooks";
import { cn } from "@/lib/utils";
import { useFetchAllChats } from "@/api/chat-messages/hooks";
import { Button } from "@/components/ui/button";
import { faCog } from "@fortawesome/free-solid-svg-icons/faCog";
import { SettingsModal } from "@/components/SettingsModal";

export const Sidebar = () => {
  return (
    <div className="flex flex-col gap-6 p-6 h-full overflow-y-auto">
      <SidebarHeader />
    </div>
  );
};

const DatasetList = () => {
  const { data } = useGeneratedDictionaries();
  const [settingsModalOpen, setSettingsModalOpen] = useState(false);

  return (
    <div className="relative flex flex-col h-full min-h-[300px]">
      <div className="flex justify-between items-center">
        <div>
          <strong>Datasets</strong>
        </div>
        <AddDataModal />
      </div>
      <div className="flex-1 flex flex-col pt-1">
        {data?.map((dictionary) => (
          <div
            key={dictionary.name}
            className="h-8 py-3 justify-start items-start gap-1 inline-flex"
          >
            <div
              className={cn("grow h-6 text-base leading-tight truncate", {
                "text-muted-foreground": dictionary.in_progress,
              })}
            >
              {dictionary.name}
            </div>
            {dictionary.in_progress && (
              <img
                src={loader}
                alt="processing"
                className="mr-2 w-4 h-4 animate-spin"
              />
            )}
          </div>
        ))}
      </div>
      <SettingsModal
        isOpen={settingsModalOpen}
        onOpenChange={setSettingsModalOpen}
      />
      <div className="mt-4 flex justify-center">
        <Button
          variant="ghost"
          size="sm"
          className="w-full flex items-center justify-center gap-2"
          onClick={() => setSettingsModalOpen(true)}
        >
          <FontAwesomeIcon icon={faCog} />
          <span>Settings</span>
        </Button>
      </div>
    </div>
  );
};

const ChatList = () => {
  const location = useLocation();
  const { data } = useFetchAllChats();
  const navigate = useNavigate();
  const chatIdMatch = location.pathname.match(/\/chats\/([^/]+)/);
  const chatId = chatIdMatch ? chatIdMatch[1] : undefined;
  const [activeKey, setActiveKey] = useState<string | undefined>(chatId);
  const [settingsModalOpen, setSettingsModalOpen] = useState(false);

  useEffect(() => {
    if (chatId) {
      setActiveKey(chatId);
    }
  }, [chatId, location.pathname]);

  const sortedChats = data?.slice().sort((a, b) => {
    const dateA = a.created_at ? new Date(a.created_at).getTime() : 0;
    const dateB = b.created_at ? new Date(b.created_at).getTime() : 0;
    return dateA - dateB;
  });

  const options: SidebarMenuOptionType[] =
    sortedChats?.map((c) => ({
      key: c.id,
      name: c.name,
      active: activeKey === c.id,
      onClick: () => {
        navigate(generateChatRoute(c.id));
      },
    })) || [];

  return (
    <div className="relative flex flex-col h-full min-h-[300px]">
      <div className="flex justify-between items-center pb-4">
        <div>
          <strong>Chats</strong>
        </div>
        <NewChatModal />
      </div>
      <div className="flex-1 overflow-y-auto">
        <SidebarMenu options={options} activeKey={activeKey} />
      </div>
      <SettingsModal
        isOpen={settingsModalOpen}
        onOpenChange={setSettingsModalOpen}
      />
      <div className="mt-4 flex justify-center">
        <Button
          variant="ghost"
          size="sm"
          className="w-full flex items-center justify-center gap-2"
          onClick={() => setSettingsModalOpen(true)}
        >
          <FontAwesomeIcon icon={faCog} />
          <span>Settings</span>
        </Button>
      </div>
    </div>
  );
};

export const Divider = () => {
  return <hr className="border-t my-4" />;
};

const SidebarHeader = () => {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const [activeKey, setActiveKey] = useState("");

  useEffect(() => {
    if (pathname.includes(ROUTES.DATA)) {
      setActiveKey(ROUTES.DATA);
    } else if (pathname.includes(ROUTES.CHATS)) {
      setActiveKey(ROUTES.CHATS);
    } else if (pathname === "/") {
      setActiveKey(ROUTES.DATA);
    }
  }, [pathname]);

  const options: SidebarMenuOptionType[] = [
    {
      key: "data",
      name: "Data",
      icon: <FontAwesomeIcon icon={faTable} />,
      active: activeKey === ROUTES.DATA,
      testId: "data-menu-option",
      onClick: () => {
        navigate(ROUTES.DATA);
      },
    },
    {
      key: "chats",
      name: "Chats",
      icon: <FontAwesomeIcon icon={faComments} />,
      active: activeKey === ROUTES.CHATS,
      testId: "chats-menu-option",
      onClick: () => {
        navigate(ROUTES.CHATS);
      },
    },
  ];

  return (
    <div className="flex flex-col gap-4 h-full">
      <img
        src={drLogo}
        alt="DataRobot"
        className="w-[130px] cursor-pointer"
        onClick={() => navigate(ROUTES.DATA)}
      />
      <h1 className="text-xl">Talk to my data</h1>
      <p className="text-sm">
        Add the data you want to analyze, then ask DataRobot questions to
        generate insights.
      </p>
      <div className="flex flex-col gap-2 flex-1 min-h-0">
        <SidebarMenu options={options} activeKey={activeKey} />
        <Separator className="my-4 border-t" />
        <WelcomeModal />
        <div className="flex-1 min-h-0">
          {activeKey === ROUTES.DATA && <DatasetList />}
          {activeKey === ROUTES.CHATS && <ChatList />}
        </div>
      </div>
    </div>
  );
};

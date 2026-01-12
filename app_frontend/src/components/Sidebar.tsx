import { useState } from 'react';
import { useTranslation } from '@/i18n';
import { useNavigate, useParams } from 'react-router-dom';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import drLogoDark from '@/assets/DataRobot_black.svg';
import drLogoLight from '@/assets/DataRobot_white.svg';
import { SidebarMenu, SidebarMenuOptionType } from '@/components/ui-custom/sidebar-menu';
import { WelcomeModal } from './WelcomeModal';
import { AddDataModal } from './AddDataModal';
import { ROUTES, generateChatRoute, generateDataRoute } from '@/pages/routes';
import { Separator } from '@radix-ui/react-separator';
import { NewChatModal } from './NewChatModal';
import { Loader2 } from 'lucide-react';
import { useGeneratedDictionaries, getDictionariesMenu } from '@/api/dictionaries';
import { useFetchAllChats, getChatsMenu } from '@/api/chat-messages';
import { Button } from '@/components/ui/button';
import { faCog } from '@fortawesome/free-solid-svg-icons/faCog';
import { SettingsModal } from '@/components/SettingsModal';
import {
  Sidebar as SidebarUI,
  SidebarProvider,
  SidebarHeader,
  SidebarContent,
  SidebarGroup,
  SidebarFooter,
} from '@/components/ui/sidebar';
import { useTheme } from '@/theme/theme-provider';

const DatasetList = ({ highlight }: { highlight: boolean }) => {
  const { data, isLoading } = useGeneratedDictionaries<SidebarMenuOptionType[]>({
    select: getDictionariesMenu,
  });
  const { t } = useTranslation();
  const params = useParams();
  const navigate = useNavigate();

  return (
    <div className="relative flex flex-col flex-1 h-full">
      <div className="flex justify-between items-center pb-3 pl-2">
        <div>
          <p className="mn-label-large">{t('Datasets')}</p>
        </div>
        <AddDataModal highlight={highlight} />
      </div>
      <div className="flex-1 overflow-y-auto">
        <SidebarMenu
          options={data}
          activeKey={params.dataId}
          onClick={({ name }) => navigate(generateDataRoute(name))}
        />
        {isLoading && (
          <div className="mt-4 flex justify-center">
            <Loader2 className="w-4 h-4 animate-spin" />
          </div>
        )}
        {!isLoading && !data?.length && (
          <p className="pl-2 text-muted-foreground">{t('Add your data here')}</p>
        )}
      </div>
    </div>
  );
};

const ChatList = ({ highlight }: { highlight: boolean }) => {
  const { data, isLoading } = useFetchAllChats<SidebarMenuOptionType[]>({ select: getChatsMenu });
  const navigate = useNavigate();
  const { chatId } = useParams();
  const { t } = useTranslation();

  return (
    <div className="relative flex flex-col flex-1 h-full">
      <div className="flex justify-between items-center pb-3 pl-2">
        <div>
          <p className="mn-label-large">{t('Chats')}</p>
        </div>
        <NewChatModal highlight={highlight} />
      </div>
      <div className="flex-1 overflow-y-auto">
        <SidebarMenu
          options={data}
          activeKey={chatId}
          onClick={({ id }) => {
            navigate(generateChatRoute(id));
          }}
        />
        {isLoading && (
          <div className="mt-4 flex justify-center">
            <Loader2 className="w-4 h-4 animate-spin" />
          </div>
        )}
        {!isLoading && !data?.length && (
          <p className="pl-2 text-muted-foreground">{t('Start your first chart here')}</p>
        )}
      </div>
    </div>
  );
};

export const Sidebar = () => {
  const { data: datasets, isLoading: isLoadingDatasets } = useGeneratedDictionaries();
  const { data: chats, isLoading: isLoadingChats } = useFetchAllChats();
  const highlightDatasets = !isLoadingDatasets && !datasets?.length;
  const highlightChats = !highlightDatasets && !isLoadingChats && !chats?.length;
  const navigate = useNavigate();
  const { theme } = useTheme();
  const drLogo = theme === 'dark' ? drLogoLight : drLogoDark;
  const { t } = useTranslation();
  const [settingsModalOpen, setSettingsModalOpen] = useState(false);

  return (
    <SidebarProvider defaultOpen={true}>
      <SidebarUI>
        <SidebarHeader>
          <img
            src={drLogo}
            alt="DataRobot"
            className="w-[130px] cursor-pointer mb-4"
            onClick={() => navigate(ROUTES.DATA)}
          />
          <h1 className="heading-04 mb-1">{t('Talk to my data')}</h1>
          <p className="body-secondary">
            {t(
              'Add the data you want to analyze, then ask DataRobot questions to generate insights.'
            )}
          </p>
        </SidebarHeader>
        <SidebarContent className="flex-none max-h-[300px]">
          <Separator className="my-6 border-t" />
          <SidebarGroup className="h-full">
            <DatasetList highlight={highlightDatasets} />
          </SidebarGroup>
        </SidebarContent>
        <Separator className="my-6 border-t" />
        <SidebarContent>
          <SidebarGroup className="h-full">
            <ChatList highlight={highlightChats} />
            <WelcomeModal />
          </SidebarGroup>
        </SidebarContent>
        <SidebarFooter>
          <SettingsModal isOpen={settingsModalOpen} onOpenChange={setSettingsModalOpen} />
          <div className="mt-4 flex justify-center">
            <Button
              variant="ghost"
              size="sm"
              className="w-full flex items-center justify-center gap-2"
              onClick={() => setSettingsModalOpen(true)}
            >
              <FontAwesomeIcon icon={faCog} />
              <span>{t('Settings')}</span>
            </Button>
          </div>
        </SidebarFooter>
      </SidebarUI>
    </SidebarProvider>
  );
};

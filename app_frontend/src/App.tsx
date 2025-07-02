import './App.css';
import { Sidebar } from '@/components/Sidebar';
import Pages from './pages';
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable';
import { useDataRobotInfo } from './api/user/hooks';

function App() {
    useDataRobotInfo();

    return (
        <div className="h-screen">
            <ResizablePanelGroup direction="horizontal">
                <ResizablePanel defaultSize={20} minSize={15} maxSize={30}>
                    <Sidebar />
                </ResizablePanel>
                <ResizableHandle withHandle />
                <ResizablePanel defaultSize={80}>
                    <Pages />
                </ResizablePanel>
            </ResizablePanelGroup>
        </div>
    );
}

export default App;

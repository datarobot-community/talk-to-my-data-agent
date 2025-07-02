import { Route, Routes, Navigate } from 'react-router-dom';
import { ROUTES } from './routes';
import { Suspense, lazy } from 'react';

const Data = lazy(() => import('./Data').then(module => ({ default: module.Data })));
const Chats = lazy(() => import('./Chats').then(module => ({ default: module.Chats })));

const Loading = () => <div className="flex items-center justify-center h-full">Loading...</div>;

const Pages = () => {
    return (
        <div className="w-full h-full overflow-y-auto">
            <Suspense fallback={<Loading />}>
                <Routes>
                    <Route path={ROUTES.DATA} element={<Data />} />
                    <Route path={ROUTES.CHATS} element={<Chats />} />
                    <Route path={ROUTES.CHAT_WITH_ID} element={<Chats />} />
                    <Route path="*" element={<Navigate to={ROUTES.DATA} replace />} />
                </Routes>
            </Suspense>
        </div>
    );
};

export default Pages;

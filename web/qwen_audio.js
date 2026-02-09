import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "Qwen.AudioLoader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        // 定义节点名称与配置的映射
        const nodeConfig = {
            "Load_Audio_Folder": {
                widgetName: "文件夹路径", // [汉化] 必须匹配 Python 中的 Key
                apiRoute: "/qwen/browse_folder",
                btnText: "📂 浏览文件夹 (Folder)"
            },
            "Load_Audio": {
                widgetName: "文件路径",   // [汉化] 必须匹配 Python 中的 Key
                apiRoute: "/qwen/browse_file",
                btnText: "🎵 浏览文件 (File)"
            }
        };

        // 检查当前节点是否在我们的配置列表中
        if (nodeConfig[nodeData.name]) {
            const config = nodeConfig[nodeData.name];
            
            // 劫持 onNodeCreated 方法
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // 查找对应的输入框 Widget
                const pathWidget = this.widgets.find((w) => w.name === config.widgetName);

                if (pathWidget) {
                    // 添加按钮
                    const btn = this.addWidget("button", config.btnText, null, () => {
                        btn.disabled = true;
                        btn.name = "选择中...";

                        api.fetchApi(config.apiRoute, { method: "POST" })
                        .then((response) => response.json())
                        .then((data) => {
                            if (data.path) {
                                pathWidget.value = data.path; // 更新输入框
                            }
                        })
                        .catch((error) => {
                            console.error("Browse Error:", error);
                            alert("打开选择框失败，请检查后台日志。");
                        })
                        .finally(() => {
                            btn.disabled = false;
                            btn.name = config.btnText;
                            app.graph.setDirtyCanvas(true, true);
                        });
                    });
                }
                return r;
            };
        }
    },
});
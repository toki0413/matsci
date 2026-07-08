/**
 * i18n — Bilingual EN/ZH translation configuration.
 *
 * Import this module in main.tsx to initialize react-i18next.
 * Use the `useTranslation()` hook in components:
 *
 *   import { useTranslation } from 'react-i18next';
 *   const { t, i18n } = useTranslation();
 *   return <span>{t('chat.welcome')}</span>;
 */
import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

const resources = {
  en: {
    translation: {
      /* ── Sidebar groups ── */
      'nav.core': 'AI CHAT',
      'nav.research': 'RESEARCH',
      'nav.workspace': 'WORKSPACE',
      'nav.system': 'SYSTEM',
      'nav.matsci': 'MATERIALS SCIENCE',

      /* ── Core tabs ── */
      'tab.chat': 'Chat',
      'tab.team': 'Team',
      'tab.coder': 'Coder',

      /* ── Research tabs ── */
      'tab.knowledge': 'Knowledge Hub',
      'tab.periodic': 'Periodic Table',
      'tab.project': 'Project',
      'tab.notebook': 'Notebook',
      'tab.benchmark': 'Benchmark',
      'tab.evolution': 'Evolution',
      'tab.execute': 'Execute',
      'tab.workflows': 'Workflows',
      'tab.sweep': 'Sweep',
      'tab.explore': 'Explore',
      'tab.diagnose': 'Diagnose',
      'tab.structure': 'Structure',
      'tab.hpc': 'HPC',

      /* ── Workspace tabs ── */
      'tab.files': 'Files',
      'tab.terminal': 'Terminal',
      'tab.sandbox': 'Sandbox',
      'tab.review': 'Review',
      'tab.tools': 'Tools',
      'tab.skills': 'Skills',

      /* ── System tabs ── */
      'tab.memory': 'Memory',
      'tab.emotion': 'Emotion',
      'tab.plugins': 'Plugins',
      'tab.threads': 'Threads',
      'tab.logs': 'Logs',
      'tab.settings': 'Settings',

      /* ── Chat panel ── */
      'chat.welcome': 'Materials Science Assistant',
      'chat.welcomeSub':
        'Ask about DFT calculations, molecular dynamics, crystal structures, or any materials science question.',
      'chat.placeholder': 'Ask Huginn anything...',
      'chat.send': 'Send',
      'chat.newChat': 'New Chat',
      'chat.copyCode': 'Copy',
      'chat.toolRunning': 'Running',
      'chat.toolDone': 'Completed',
      'chat.toolError': 'Error',
      'chat.search': 'Search messages',
      'chat.mode.chat': 'Chat',
      'chat.mode.plan': 'Plan',
      'chat.mode.build': 'Build',
      'chat.mode.chat.desc': 'Normal assistant chat',
      'chat.mode.plan.desc': 'Generate a step-by-step plan without executing tools',
      'chat.mode.build.desc': 'Execute tools and edit files',

      /* ── Language switcher ── */
      'lang.label': 'EN',
      'lang.switchTo': 'Switch to Chinese',

      /* ── Status / connection bar ── */
      'status.connected': 'Connected',
      'status.connecting': 'Connecting',
      'status.disconnected': 'Disconnected',
      'status.model': 'Model',
      'status.agent': 'Agent',
      'status.tools': 'Tools',

      /* ── Knowledge base panel ── */
      'kb.searchPlaceholder': 'Search documents...',
      'kb.upload': 'Upload Document',
      'kb.noResults': 'No matching documents',
      'kb.docCount': 'documents',
      'kb.deepParse': 'Deep Parse',
      'kb.graph': 'Document Graph',

      /* ── Knowledge Hub — Topics ── */
      'topic.title': 'Research Topics',
      'topic.new': 'New Topic',
      'topic.docs': 'documents',
      'topic.notes': 'notes',
      'topic.chatSaves': 'saved from chat',
      'topic.lastUpdated': 'Updated',
      'topic.searchPlaceholder': 'Search topics...',
      'topic.empty': 'No topics yet. Create one to organize your research.',
      'topic.back': 'All Topics',
      'topic.source.upload': 'Uploaded',
      'topic.source.chat': 'From Chat',
      'topic.source.note': 'Note',

      /* ── Depth mode ── */
      'depth.label': 'Depth',
      'depth.quick': 'Quick',
      'depth.deep': 'Deep',
      'depth.research': 'Research',
      'depth.quick.desc': 'Brief answers',
      'depth.deep.desc': 'Detailed analysis',
      'depth.research.desc': 'Full literature review',

      /* ── Structured output ── */
      'output.label': 'Output',
      'output.text': 'Text',
      'output.outline': 'Outline',
      'output.mindmap': 'Mindmap',
      'output.table': 'Table',
      'output.outline.l1.params': 'VASP Input File Parameters',
      'output.outline.l2.poscar': 'POSCAR: Crystal structure definition',
      'output.outline.l2.incar': 'INCAR: Calculation control flags',
      'output.outline.l1.convergence': 'Convergence Criteria',
      'output.outline.l2.encut': 'ENCUT: 520 eV cutoff energy',
      'output.outline.l2.kpoints': 'K-points: 11×11×11 Monkhorst-Pack',
      'output.outline.l1.results': 'Results & Analysis',
      'output.outline.l2.energy': 'Energy: -8.312 eV/atom (converged)',
      'output.outline.l2.magmom': 'Magnetic moment: 2.22 μB',
      'output.mindmap.center': 'DFT Workflow',
      'output.mindmap.converged': 'Converged',
      'output.table.header.parameter': 'Parameter',
      'output.table.header.value': 'Value',
      'output.table.header.status': 'Status',
      'output.table.functional': 'Functional',
      'output.table.standard': 'Standard',
      'output.table.converged': 'Converged',
      'output.table.tight': 'Tight',
      'output.table.fullRelax': 'Full relax',

      /* ── Save to topic ── */
      'save.toTopic': 'Save to topic',
      'save.selectTopic': 'Select a topic',
      'save.saved': 'Saved',
      'save.newTopic': '+ New topic',
      'save.toMemory': 'Save to memory',
      'save.confirm': 'Save',

      /* ── Pet widget ── */
      'pet.xp': 'Experience',
      'pet.hunger': 'Hunger',
      'pet.mood': 'Mood',
      'pet.online': 'Online',
      'pet.offline': 'Offline',
      'pet.feed': 'Feed',
      'pet.play': 'Play',

      /* ── Suggestion prompts ── */
      'suggest.dft': 'Run DFT relaxation for BCC iron',
      'suggest.md': 'Set up LAMMPS MD simulation for copper',
      'suggest.band': 'Calculate band structure of silicon',
      'suggest.xrd': 'Predict XRD pattern for TiO2 rutile',

      /* ── Settings panel ── */
      'settings.title': 'Settings',
      'settings.general': 'General',
      'settings.models': 'Models',
      'settings.agents': 'Agents',
      'settings.privacy': 'Privacy',
      'settings.pet': 'Pet',
      'settings.security': 'Security',
      'settings.credentials': 'Credentials',
      'settings.jobs': 'Jobs',
      'settings.export': 'Export',
      'settings.bot': 'Bot',

      /* ── Computation card ── */
      'card.preview': 'Preview',
      'card.ready': 'Ready to submit',
      'card.submitted': 'Submitted',

      /* ── Job status card ── */
      'job.running': 'Running',
      'job.completed': 'Completed',
      'job.failed': 'Failed',
      'job.queue': 'Queue',
      'job.nodes': 'Nodes',
      'job.elapsed': 'Elapsed',
      'job.eta': 'ETA',

      /* ── Result chart ── */
      'result.energy': 'Energy',
      'result.magMom': 'Mag. Moment',
      'result.volume': 'Volume',
      'result.status': 'Status',
      'result.converged': 'Converged',
      'result.notConverged': 'Not converged',
      'result.relevance': 'match',
      'result.outputFiles': 'Output files',

      /* ── Panel descriptions ── */
      'panel.team.desc': 'Multi-agent collaboration for complex research workflows',
      'panel.coder.desc': 'Autonomous coding for materials science scripts',
      'panel.periodic.desc': 'Interactive periodic table with element properties',
      'panel.structure.desc': '3D crystal structure viewer and analyzer',
      'panel.settings.desc': 'Configure Huginn preferences and integrations',

      /* ── Empty states ── */
      'empty.noData': 'No data available',
      'empty.loading': 'Loading...',
      'empty.error': 'An error occurred',
    },
  },
  zh: {
    translation: {
      /* ── Sidebar groups ── */
      'nav.core': 'AI 对话',
      'nav.research': '研究',
      'nav.workspace': '工作区',
      'nav.system': '系统',
      'nav.matsci': '材料科学',

      /* ── Core tabs ── */
      'tab.chat': '对话',
      'tab.team': '团队',
      'tab.coder': '编程',

      /* ── Research tabs ── */
      'tab.knowledge': '知识中心',
      'tab.periodic': '元素周期表',
      'tab.project': '项目',
      'tab.notebook': '笔记本',
      'tab.benchmark': '基准测试',
      'tab.evolution': '进化',
      'tab.execute': '执行',
      'tab.workflows': '工作流',
      'tab.sweep': '参数扫描',
      'tab.explore': '探索',
      'tab.diagnose': '诊断',
      'tab.structure': '结构查看',
      'tab.hpc': '超算',

      /* ── Workspace tabs ── */
      'tab.files': '文件',
      'tab.terminal': '终端',
      'tab.sandbox': '沙箱',
      'tab.review': '审查',
      'tab.tools': '工具',
      'tab.skills': '技能',

      /* ── System tabs ── */
      'tab.memory': '记忆',
      'tab.emotion': '情绪',
      'tab.plugins': '插件',
      'tab.threads': '对话线',
      'tab.logs': '日志',
      'tab.settings': '设置',

      /* ── Chat panel ── */
      'chat.welcome': '材料科学助手',
      'chat.welcomeSub': '可以询问 DFT 计算、分子动力学、晶体结构，或任何材料科学问题。',
      'chat.placeholder': '向 Huginn 提问...',
      'chat.send': '发送',
      'chat.newChat': '新对话',
      'chat.copyCode': '复制',
      'chat.toolRunning': '运行中',
      'chat.toolDone': '已完成',
      'chat.toolError': '出错',
      'chat.search': '搜索消息',
      'chat.mode.chat': '对话',
      'chat.mode.plan': '规划',
      'chat.mode.build': '构建',
      'chat.mode.chat.desc': '常规助手对话',
      'chat.mode.plan.desc': '生成分步计划，不执行工具',
      'chat.mode.build.desc': '执行工具并编辑文件',

      /* ── Language switcher ── */
      'lang.label': '中',
      'lang.switchTo': 'Switch to English',

      /* ── Status / connection bar ── */
      'status.connected': '已连接',
      'status.connecting': '连接中',
      'status.disconnected': '未连接',
      'status.model': '模型',
      'status.agent': '智能体',
      'status.tools': '工具',

      /* ── Knowledge base panel ── */
      'kb.searchPlaceholder': '搜索文档...',
      'kb.upload': '上传文档',
      'kb.noResults': '没有匹配的文档',
      'kb.docCount': '篇文档',
      'kb.deepParse': '深度解析',
      'kb.graph': '文档图谱',

      /* ── Knowledge Hub — Topics ── */
      'topic.title': '研究专题',
      'topic.new': '新建专题',
      'topic.docs': '篇文档',
      'topic.notes': '条笔记',
      'topic.chatSaves': '条对话收藏',
      'topic.lastUpdated': '更新于',
      'topic.searchPlaceholder': '搜索专题...',
      'topic.empty': '还没有专题。创建一个来整理你的研究。',
      'topic.back': '所有专题',
      'topic.source.upload': '上传',
      'topic.source.chat': '来自对话',
      'topic.source.note': '笔记',

      /* ── Depth mode ── */
      'depth.label': '深度',
      'depth.quick': '简洁',
      'depth.deep': '深入',
      'depth.research': '研究',
      'depth.quick.desc': '简要回答',
      'depth.deep.desc': '详细分析',
      'depth.research.desc': '全面文献综述',

      /* ── Structured output ── */
      'output.label': '输出',
      'output.text': '文本',
      'output.outline': '大纲',
      'output.mindmap': '思维导图',
      'output.table': '表格',
      'output.outline.l1.params': 'VASP 输入文件参数',
      'output.outline.l2.poscar': 'POSCAR: 晶体结构定义',
      'output.outline.l2.incar': 'INCAR: 计算控制标志',
      'output.outline.l1.convergence': '收敛标准',
      'output.outline.l2.encut': 'ENCUT: 520 eV 截断能',
      'output.outline.l2.kpoints': 'K 点: 11×11×11 Monkhorst-Pack',
      'output.outline.l1.results': '结果与分析',
      'output.outline.l2.energy': '能量: -8.312 eV/atom（已收敛）',
      'output.outline.l2.magmom': '磁矩: 2.22 μB',
      'output.mindmap.center': 'DFT 工作流',
      'output.mindmap.converged': '已收敛',
      'output.table.header.parameter': '参数',
      'output.table.header.value': '值',
      'output.table.header.status': '状态',
      'output.table.functional': '泛函',
      'output.table.standard': '标准',
      'output.table.converged': '已收敛',
      'output.table.tight': '严格',
      'output.table.fullRelax': '完全弛豫',

      /* ── Save to topic ── */
      'save.toTopic': '收藏到专题',
      'save.selectTopic': '选择专题',
      'save.saved': '已保存',
      'save.newTopic': '+ 新建专题',
      'save.toMemory': '保存到记忆',
      'save.confirm': '保存',

      /* ── Pet widget ── */
      'pet.xp': '经验',
      'pet.hunger': '饥饿度',
      'pet.mood': '心情',
      'pet.online': '在线',
      'pet.offline': '离线',
      'pet.feed': '喂食',
      'pet.play': '玩耍',

      /* ── Suggestion prompts ── */
      'suggest.dft': '对 BCC 铁进行 DFT 结构优化',
      'suggest.md': '设置铜的 LAMMPS 分子动力学模拟',
      'suggest.band': '计算硅的能带结构',
      'suggest.xrd': '预测金红石 TiO2 的 XRD 图谱',

      /* ── Settings panel ── */
      'settings.title': '设置',
      'settings.general': '通用',
      'settings.models': '模型',
      'settings.agents': '智能体',
      'settings.privacy': '隐私',
      'settings.pet': '宠物',
      'settings.security': '安全',
      'settings.credentials': '凭据',
      'settings.jobs': '任务',
      'settings.export': '导出',
      'settings.bot': '机器人',

      /* ── Computation card ── */
      'card.preview': '预览',
      'card.ready': '可提交',
      'card.submitted': '已提交',

      /* ── Job status card ── */
      'job.running': '运行中',
      'job.completed': '已完成',
      'job.failed': '失败',
      'job.queue': '队列',
      'job.nodes': '节点',
      'job.elapsed': '已用时',
      'job.eta': '剩余时间',

      /* ── Result chart ── */
      'result.energy': '能量',
      'result.magMom': '磁矩',
      'result.volume': '体积',
      'result.status': '状态',
      'result.converged': '已收敛',
      'result.notConverged': '未收敛',
      'result.relevance': '匹配',
      'result.outputFiles': '输出文件',

      /* ── Panel descriptions ── */
      'panel.team.desc': '多智能体协同处理复杂研究工作流',
      'panel.coder.desc': '自主编写材料科学计算脚本',
      'panel.periodic.desc': '交互式元素周期表与元素属性',
      'panel.structure.desc': '3D 晶体结构查看与分析',
      'panel.settings.desc': '配置 Huginn 偏好和集成',

      /* ── Empty states ── */
      'empty.noData': '暂无数据',
      'empty.loading': '加载中...',
      'empty.error': '发生错误',
    },
  },
};

i18n.use(initReactI18next).init({
  resources,
  lng: 'en',
  fallbackLng: 'en',
  interpolation: { escapeValue: false },
});

export default i18n;

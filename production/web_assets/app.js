'use strict';
const $ = id => document.getElementById(id);
const number = value => Number.isFinite(value) ? new Intl.NumberFormat('zh-CN', {maximumFractionDigits:0}).format(value) : '—';
const clock = (ms, date=false) => ms ? new Intl.DateTimeFormat('zh-CN', {timeZone:'Asia/Shanghai', ...(date?{month:'2-digit',day:'2-digit'}:{}), hour:'2-digit', minute:'2-digit', hour12:false}).format(ms) : '—';
let snapshot = null, selectedRank = null, selectedEvent = null, historyVersion = 0, loading = false, pendingRefresh = false, lastPoints = [], lastCurve = [], lastTerminal = null, lastCheckpoints = [], clockOffset = 0, pendingSelection = null;
const calendar = ms => new Intl.DateTimeFormat('zh-CN', {timeZone:'Asia/Shanghai',year:'numeric',month:'long',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false}).format(ms);
const dayLabel = ms => new Intl.DateTimeFormat('zh-CN', {timeZone:'Asia/Shanghai',month:'2-digit',day:'2-digit'}).format(ms);

function currentEvent() {
  return snapshot?.events?.find(event=>event.event===selectedEvent)
    || snapshot?.predictions.find(row=>row.event===selectedEvent&&row.start_ms&&row.end_ms);
}

function historicalMode(){return currentEvent()?.end_ms<=Date.now()+clockOffset;}
const hasObservation=row=>row?.observation_available??row?.available??false;
const observedValue=row=>Number.isFinite(row?.actual_final_ep)?row.actual_final_ep:row?.current_ep;

function startSelectionSwitch(){pendingSelection=`${selectedEvent}:${selectedRank}`;ForecastUI.startView(document.querySelector('.view-switch-region'));}
function finishSelectionSwitch(){
  if(pendingSelection===`${selectedEvent}:${selectedRank}`){pendingSelection=null;ForecastUI.finishView(document.querySelector('.view-switch-region'));}
}
function revealSelectedRow(){
  const row=document.querySelector('#rank-rows tr.selected'),container=document.querySelector('.ranking .table-wrap');
  if(!row||!container)return;
  const a=row.getBoundingClientRect(),b=container.getBoundingClientRect(),header=container.querySelector('thead').getBoundingClientRect().height;
  if(a.top<b.top+header)container.scrollTop-=b.top+header-a.top;
  else if(a.bottom>b.bottom)container.scrollTop+=a.bottom-b.bottom;
}

function renderEvent() {
  const event = currentEvent();
  $('event-id').textContent=selectedEvent ? `活动 ID · ${selectedEvent}` : '';
  $('event-id').hidden=!selectedEvent;
  $('event-name').textContent = event?.name || (selectedEvent ? `活动 ${selectedEvent}` : '当前没有进行中的国服活动');
  $('event-type').textContent = event?.type_label || '';
  $('event-cn-start').textContent = event?.start_ms ? calendar(event.start_ms)+' 开始（CN）' : '';
  $('event-cn-end').textContent = event?.end_ms ? calendar(event.end_ms)+' 结束（CN）' : '';
  $('event-jp-start').textContent = event?.jp_start_ms ? calendar(event.jp_start_ms)+' 开始（JP）' : '';
  $('event-jp-start').hidden = !event?.jp_start_ms;
  const image = $('event-banner');
  image.parentElement.hidden=!event?.banner_url;
  if (event?.banner_url) {
    if (image.getAttribute('src') !== event.banner_url) {
      image.parentElement.dataset.placeholder='正在加载活动横幅…';
      image.hidden=true; image.alt=(event.name||'')+' 活动横幅'; image.src=event.banner_url;
    }
  } else {image.hidden=true; image.removeAttribute('src');}
  $('chart-title').textContent = event ? `${event.name||'活动 '+selectedEvent} (CN T${selectedRank})` : '等待活动信息';
  renderCountdown();
}

function renderCountdown() {
  const event = currentEvent();
  if (!event?.start_ms || !event?.end_ms) {
    $('event-progress').textContent='—'; $('event-remaining').textContent='等待活动时间'; return;
  }
  const now=Date.now()+clockOffset, progress=Math.max(0,Math.min(1,(now-event.start_ms)/(event.end_ms-event.start_ms)));
  $('event-progress').textContent=`${(progress*100).toFixed(1)}% 完成`;
  if(now>=event.end_ms){$('event-remaining').textContent='活动已结束';return;}
  if(now<event.start_ms){$('event-remaining').textContent='活动尚未开始';return;}
  let seconds=Math.max(0,Math.floor((event.end_ms-now)/1000));
  const days=Math.floor(seconds/86400);seconds%=86400;
  const hours=Math.floor(seconds/3600);seconds%=3600;
  const minutes=Math.floor(seconds/60);seconds%=60;
  $('event-remaining').textContent=`距结束 ${days}天 ${hours}小时 ${minutes}分 ${seconds}秒`;
}

async function getJSON(url) {
  const response = await fetch(url, {cache:'no-store'});
  if (!response.ok) throw new Error('暂时无法读取结果，请稍后刷新');
  return response.json();
}

function notice(text) {
  $('notice').textContent = text;
  $('notice').hidden = !text;
}

function render() {
  if (!snapshot) return;
  const replay=snapshot.service?.mode==='replay';
  $('replay-banner').hidden=!replay;
  if(replay)$('replay-banner').textContent=`历史逐步回放 · 模拟时间 ${calendar(snapshot.server_time_ms)} · ${snapshot.service.frames_completed}/${snapshot.service.total_frames} 步 · 训练 ${snapshot.service.training_events} 期。${snapshot.service.note||''}${snapshot.service.error?' 回放错误：'+snapshot.service.error:''}`;
  ForecastUI.sync();
  const rows = snapshot.predictions.filter(row => row.event === selectedEvent);
  const row = rows.find(item => item.rank === selectedRank);
  const historical=historicalMode();
  renderEvent();
  $('observation-label').textContent=historical?(Number.isFinite(row?.actual_final_ep)?'实际终分':'最后观测分数'):'当前分数';
  $('forecast-label').textContent=historical?'历史最后预测':'预测终分';
  $('table-observation-label').textContent=historical?'终分 / 最后观测':'当前分数';
  $('table-forecast-label').textContent=historical?'历史最后预测':'预测终分';
  $('history-title').textContent=historical?'历史活动曲线':'本期预测记录';
  document.querySelector('.observed-key').textContent=historical?'历史分数线':'当前分数线';
  document.querySelector('.predicted-key').textContent=historical?'历史终分预测':'预测终分';
  $('current-score').textContent = hasObservation(row) ? number(observedValue(row)) : '—';
  $('predicted-score').textContent = row?.available ? number(row.predicted_final_ep) : '—';
  $('observed-time').textContent = row?.observed_at ? '观测于 '+clock(row.observed_at, true) : '等待观测数据';
  $('prediction-detail').textContent = row?.available ? (historical?'预测于 '+clock(row.forecast_issued_at,true):`T${selectedRank} · 终分预测${row.provisional?' · 暂定预测':''}`) : (row?.reason || '等待预测数据');
  const messages = [];
  if (!historical&&snapshot.stale) messages.push('显示上次保存的结果，数据尚未更新。');
  if (!historical&&snapshot.data_update_failed) messages.push('本轮更新有异常，仍有效的缓存观测可继续预测。');
  if (!historical&&snapshot.service.state === 'error') messages.push('预测更新失败，将在下一周期重试。');
  if (!historical&&snapshot.service.offline) messages.push('离线模式：使用本地缓存观测。');
  notice(messages.join(' '));
  const updating = ['starting','updating'].includes(snapshot.service.state);
  $('worker-state').textContent = historical?'历史活动 · 已保存数据':updating ? '正在取数并更新预测…' : snapshot.service.next_update_ms ? '最近取数 '+clock(snapshot.as_of_ms)+' · 下次取数 '+clock(snapshot.service.next_update_ms) : '';
  $('cadence').textContent = historical?'历史活动':`每${snapshot.interval_seconds/60}分钟取数`;
  const lastObserved=Math.max(0,...rows.map(value=>value.observed_at||0));
  $('as-of').textContent = historical?(lastObserved?'历史记录截至 '+clock(lastObserved,true)+'（北京时间）':'历史活动 · 暂无观测记录'):snapshot.as_of_ms ? '结果更新于 '+clock(snapshot.as_of_ms,true)+'（北京时间）' : '等待首次更新';
  const body = $('rank-rows'); body.replaceChildren();
  $('rank-count').textContent=snapshot.ranks.length+'条分数线';
  for (const rank of snapshot.ranks) {
    const value = rows.find(item => item.rank === rank);
    const tr = document.createElement('tr');tr.dataset.rank=String(rank);
    if (rank === selectedRank) tr.className = 'selected';
    const label = document.createElement('td'), button = document.createElement('button');
    button.type = 'button'; button.textContent = 'T'+rank;
    button.className='ui-button ui-button-rank';button.setAttribute('aria-pressed',String(rank===selectedRank));
    tr.addEventListener('click',event=>{chooseRank(rank);if(event.detail===0)document.querySelector(`#rank-rows tr[data-rank="${rank}"] button`)?.focus({preventScroll:true});});
    label.append(button); tr.append(label);
    const status=historical?(Number.isFinite(value?.relative_error_pct)?'误差 '+ForecastChart.formatError(value.relative_error_pct):Number.isFinite(value?.actual_final_ep)?'已核实终分':hasObservation(value)?'终分未核实':'缺少曲线'):!value?.available?(value?.reason||'等待数据'):value.provisional?'历史支持较少':snapshot.stale?'上次保存的结果':'可用';
    for (const [text,meta,metaLabel] of [[hasObservation(value)?number(observedValue(value)):'—',clock(value?.observed_at,historical),'观测时间'],[value?.available?number(value.predicted_final_ep):'—',status,'状态']]) {
      const td = document.createElement('td'),main=document.createElement('span'),detail=document.createElement('span');
      main.className='rank-cell-value';main.textContent=text;
      detail.className='rank-cell-meta';detail.textContent=meta;detail.setAttribute('aria-label',metaLabel+'：'+meta);
      td.append(main,detail);tr.append(td);
    }
    body.append(tr);
  }
}

function chooseRank(rank) {
  if(rank===selectedRank)return;
  selectedRank = rank;startSelectionSwitch();$('rank-select').value = String(rank); render();revealSelectedRow();drawHistory([]);
  $('history-detail').textContent='正在读取本期记录……'; loadHistory();
}

async function refresh() {
  if (loading){pendingRefresh=true;return;}
  loading = true; $('refresh').disabled = true;$('refresh').classList.add('is-loading');$('refresh').setAttribute('aria-busy','true');
  try {
    const requested=selectedEvent;
    const received=await getJSON('/api/predictions'+(requested?'?event='+requested:''));
    if(selectedEvent!==requested){pendingRefresh=true;return;}
    snapshot=received;
    clockOffset = snapshot.server_time_ms-Date.now();
    const select = $('rank-select');
    select.replaceChildren(...snapshot.ranks.map(rank => new Option('T'+rank, String(rank))));
    selectedRank = snapshot.ranks.includes(selectedRank) ? selectedRank : snapshot.ranks.includes(snapshot.initial_rank) ? snapshot.initial_rank : snapshot.ranks[0];
    select.value = String(selectedRank);
    const events = [...new Set([...(snapshot.events||[]).map(event=>event.event),...snapshot.active_events, ...snapshot.predictions.map(row=>row.event)])];
    selectedEvent = events.includes(selectedEvent) ? selectedEvent : snapshot.selected_event || snapshot.active_events[0] || events[0] || null;
    if(selectedEvent&&!snapshot.predictions.some(row=>row.event===selectedEvent)){
      const target=selectedEvent,details=await getJSON('/api/predictions?event='+target);
      if(selectedEvent!==target){pendingRefresh=true;return;}
      snapshot=details;
    }
    $('event-select').replaceChildren(...events.map(event => new Option(`${event} · ${snapshot.events?.find(item=>item.event===event)?.name || '活动标题待更新'}`, String(event))));
    $('event-select').value = String(selectedEvent);
    select.disabled=!snapshot.ranks.length;$('event-select').disabled=!events.length;
    render(); await loadHistory();
  } catch (error) {
    notice(error.message+'。页面保留上次读取的内容。');
    $('worker-state').textContent = '连接失败';
    if(!pendingRefresh)finishSelectionSwitch();
  } finally {
    loading = false; $('refresh').disabled = false;$('refresh').classList.remove('is-loading');$('refresh').removeAttribute('aria-busy');
    if(pendingRefresh){pendingRefresh=false;queueMicrotask(refresh);}
  }
}

function svgElement(tag, attributes, text) {
  const element = document.createElementNS('http://www.w3.org/2000/svg',tag);
  for (const [key,value] of Object.entries(attributes)) element.setAttribute(key,String(value));
  if (text !== undefined) element.textContent = text;
  return element;
}

function renderCheckpoints(checkpoints,finalScore){
  const comparison=$('checkpoint-comparison');
  comparison.hidden=!historicalMode()||!checkpoints.length;
  const body=$('checkpoint-rows');body.replaceChildren();
  for(const point of checkpoints){
    const tr=document.createElement('tr');tr.dataset.checkpoint=point.key;
    const label=document.createElement('td');
    const name=document.createElement('span');name.textContent=point.label;label.append(name);
    const hint=document.createElement('span');hint.className='checkpoint-meta';
    hint.textContent=point.available?'目标 '+clock(point.target_ms,true):point.reason;label.append(hint);tr.append(label);
    const value=document.createElement('td');
    const score=document.createElement('span');score.textContent=point.available?number(point.predicted_final_ep):'—';value.append(score);
    if(point.available){const issued=document.createElement('span');issued.className='checkpoint-meta';issued.textContent=clock(point.issued_at,true);value.append(issued);}
    tr.append(value);
    const error=document.createElement('td');error.textContent=point.available&&finalScore!==null?
      ForecastChart.formatError(ForecastChart.errorPercent(point.predicted_final_ep,finalScore)):'—';tr.append(error);
    if(point.available){
      tr.tabIndex=0;tr.setAttribute('role','button');tr.setAttribute('aria-label',`${point.label}，预测 ${number(point.predicted_final_ep)}，误差 ${error.textContent}`);
      const inspect=()=>ForecastChart.inspectCheckpoint(point);
      tr.addEventListener('click',inspect);tr.addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();inspect();}});
    }
    body.append(tr);
  }
}

function drawHistory(points, curve=[], terminal=null, checkpoints=[]) {
  const svg = $('history-chart'); ForecastChart.reset(svg); svg.replaceChildren();
  lastPoints = points; lastCurve = curve; lastTerminal = terminal;lastCheckpoints=checkpoints;
  const finalScore=historicalMode()&&Number.isFinite(terminal?.ep)&&terminal.ep>0?terminal.ep:null;
  const verification=$('forecast-verification');
  verification.hidden=!historicalMode();
  verification.textContent=finalScore===null?'活动已结束 · 等待终分核实':
    !points.length?`实际终分 ${number(finalScore)} · 没有保存的预测，无法核对`:
    `实际终分 ${number(finalScore)} · ${checkpoints.filter(point=>point.available).length} / ${checkpoints.length} 个时点有预测记录`;
  renderCheckpoints(checkpoints,finalScore);
  document.querySelector('.final-key').hidden=finalScore===null;
  $('chart-empty').textContent = historicalMode()?'该活动没有此分数线的本地曲线':'有记录后显示曲线';
  const event = currentEvent();
  if (!event || !Number.isFinite(event.start_ms) || !Number.isFinite(event.end_ms) || event.end_ms<=event.start_ms) {
    $('chart-empty').hidden = false;
    $('chart-empty').textContent = '等待活动起止时间';
    return;
  }
  const first = event.start_ms, last = event.end_ms;
  points = points.filter(point => point.observed_at >= first && point.observed_at <= last);
  const observed = (curve.length?curve:points.map(point=>({time:point.observed_at,ep:point.current_ep})))
    .filter(point=>point.time>=first&&point.time<=last);
  $('chart-empty').hidden = points.length+observed.length > 0;
  const width = Math.max(300, svg.clientWidth), height = svg.clientHeight;
  const left = width<600?58:72, right = width-16, bottom = height-40, top=20;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const all = [...observed.map(point=>point.ep), ...points.map(point=>point.predicted_final_ep),
    ...checkpoints.filter(point=>point.available).map(point=>point.predicted_final_ep),...(finalScore===null?[]:[finalScore])];
  const maximum=all.length?Math.max(...all):1, rawStep=Math.max(1,maximum)/8;
  const power=10**Math.floor(Math.log10(rawStep));
  const step=[1,2,2.5,5,10].map(value=>value*power).find(value=>value>=rawStep);
  const high=Math.max(step,Math.ceil(maximum/step)*step), ticks=Math.round(high/step);
  const x = value => left+(value-first)/(last-first)*(right-left);
  const y = value => bottom-value/high*(bottom-top);
  const unitNumber=value=>new Intl.NumberFormat('zh-CN',{maximumFractionDigits:1}).format(value);
  const compact=value=>value>=1e8 ? unitNumber(value/1e8)+'亿' : value>=1e4 ? unitNumber(value/1e4)+'万' : number(value);
  for (let i=0;i<=ticks;i++) {
    const value = step*i, yy=y(value);
    svg.append(svgElement('line',{x1:left,y1:yy,x2:right,y2:yy,class:'grid'}));
    svg.append(svgElement('text',{x:left-10,y:yy+4,'text-anchor':'end'},compact(value)));
  }
  const day=86400000, offset=8*3600000;
  const days=[];
  for(let time=Math.floor((first+offset)/day)*day-offset+day;time<last;time+=day)days.push(time);
  for(const time of [first,...days,last])svg.append(svgElement('line',{x1:x(time),y1:top,x2:x(time),y2:bottom,class:'grid'}));
  svg.append(svgElement('line',{x1:left,y1:bottom,x2:right,y2:bottom,class:'axis'}));
  svg.append(svgElement('text',{x:left,y:height-12,'text-anchor':'start'},clock(first,true)));
  let lastLabel=left;
  for(const time of days){
    const xx=x(time);
    const edgeGap=width<600?105:125;
    if(xx-left>edgeGap&&right-xx>edgeGap&&xx-lastLabel>85){
      svg.append(svgElement('text',{x:xx,y:height-12,'text-anchor':'middle'},dayLabel(time)));lastLabel=xx;
    }
  }
  svg.append(svgElement('text',{x:right,y:height-12,'text-anchor':'end'},clock(last,true)));
  if(observed.length){
    const coords=observed.map(point=>`${x(point.time)},${y(point.ep)}`).join(' ');
    svg.append(svgElement('polygon',{points:`${x(observed[0].time)},${bottom} ${coords} ${x(observed.at(-1).time)},${bottom}`,class:'observed-area'}));
    svg.append(svgElement('polyline',{points:coords,class:'observed'}));
  }
  if(finalScore!==null){
    svg.append(svgElement('line',{x1:left,y1:y(finalScore),x2:right,y2:y(finalScore),class:'actual-final'}));
  }
  if(points.length){
    svg.append(svgElement('polyline',{points:points.map(point=>`${x(point.observed_at)},${y(point.predicted_final_ep)}`).join(' '),class:'predicted'}));
    const latest=points.at(-1);
    svg.append(svgElement('line',{x1:x(latest.observed_at),y1:y(latest.predicted_final_ep),x2:right,y2:y(latest.predicted_final_ep),class:'predicted projected'}));
    svg.append(svgElement('circle',{cx:x(latest.observed_at),cy:y(latest.predicted_final_ep),r:3.5,fill:'#2563eb',class:'chart-last-point'}));
  }
  if(historicalMode()){
    const unique=new Map();
    for(const point of checkpoints.filter(point=>point.available)){
      const key=`${point.observed_at}:${point.issued_at}`;
      if(unique.has(key)){
        const shared=unique.get(key);shared.label+=' / '+point.label;
        if(shared.target_ms!==point.target_ms)shared.target_ms=null;
      }
      else unique.set(key,{...point});
    }
    const occupied=[];
    for(const point of unique.values()){
      const xx=x(point.observed_at),yy=y(point.predicted_final_ep);
      const error=ForecastChart.errorPercent(point.predicted_final_ep,finalScore);
      const marker=svgElement('g',{class:'checkpoint-marker',tabindex:0,role:'button',
        'aria-label':`${point.label}，预测 ${number(point.predicted_final_ep)}，误差 ${ForecastChart.formatError(error)}`});
      marker.append(svgElement('circle',{cx:xx,cy:yy,r:4.5}));
      if(error!==null){
        const cx=Math.max(left+36,Math.min(right-36,xx));let labelY=yy-18;
        for(const offset of [-18,24,-40,46,-62,68,-84,90]){
          const candidate=Math.max(top+14,Math.min(bottom-10,yy+offset));
          if(!occupied.some(r=>Math.abs(r.x-cx)<74&&Math.abs(r.y-candidate)<22)){labelY=candidate;break;}
        }
        occupied.push({x:cx,y:labelY});
        marker.prepend(svgElement('line',{x1:xx,y1:yy,x2:cx,y2:labelY+(labelY<yy?7:-13),class:'checkpoint-link'}));
        marker.append(svgElement('rect',{x:cx-35,y:labelY-13,width:70,height:20,rx:4}));
        marker.append(svgElement('text',{x:cx,y:labelY+1,'text-anchor':'middle'},ForecastChart.formatError(error)));
      }
      const inspect=event=>{event.stopPropagation();ForecastChart.inspectCheckpoint(point);};
      marker.addEventListener('pointermove',inspect);marker.addEventListener('pointerdown',inspect);
      marker.addEventListener('focus',()=>ForecastChart.inspectCheckpoint(point));
      marker.addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();inspect(event);}});
      svg.append(marker);
    }
  }
  if(observed.length||points.length)ForecastChart.attach({svg,first,last,left,right,top,bottom,x,y,observed,predicted:points,finalScore,formatTime:time=>clock(time,true),formatValue:number,projectionLabel:historicalMode()?'历史预测延伸':'当前预测延伸'});
}

async function loadHistory() {
  const version = ++historyVersion;
  if (!selectedEvent || !selectedRank) {drawHistory([]);finishSelectionSwitch();return;}
  try {
    const query=`event=${selectedEvent}&rank=${selectedRank}${snapshot?.service?.mode==='replay'?'&as_of='+snapshot.server_time_ms:''}`;
    const [history,actual]=await Promise.allSettled([getJSON('/api/history?'+query),getJSON('/api/curve?'+query)]);
    if (version !== historyVersion) return;
    const points = (history.status==='fulfilled'?history.value.points:[]).filter(point=>Number.isFinite(point.current_ep)&&Number.isFinite(point.predicted_final_ep));
    let curve = (actual.status==='fulfilled'?actual.value.points:[]).filter(point=>Number.isFinite(point.time)&&Number.isFinite(point.ep));
    const terminal=actual.status==='fulfilled'?actual.value.terminal_score:null;
    const checkpoints=history.status==='fulfilled'?(history.value.checkpoints||[]):[];
    if(terminal&&historicalMode())curve=[...curve.filter(point=>point.time<terminal.event_end_ms),{time:terminal.event_end_ms,ep:terminal.ep}];
    document.querySelector('.predicted-key').hidden=!points.length;
    $('history-detail').textContent = `${curve.length}个分数点 · ${points.length}条上线预测记录。${!points.length?'该活动没有保存的上线预测。':'悬停或点选查看读数，虚线后段为最后预测延伸。'}${terminal&&historicalMode()?'终点为已核实终分，上报于'+clock(terminal.reported_at,true)+'。相对误差＝（预测－终分）÷终分；正值为高估，负值为低估。':''}时间均为北京时间。`;
    if(history.status==='rejected'||actual.status==='rejected')$('history-detail').textContent+=' 部分曲线读取失败，请稍后刷新。';
    $('history-detail').textContent+=' 阶段核对只采用目标时点前1小时内已发布的记录；末次预测单列，缺失时不补算。';
    drawHistory(points,curve,terminal,checkpoints);
  } catch {
    if (version !== historyVersion) return;
    drawHistory([]); $('chart-empty').textContent = '暂时无法读取预测记录';
  } finally {
    if(version===historyVersion)finishSelectionSwitch();
  }
}

$('rank-select').addEventListener('change',event=>chooseRank(Number(event.target.value)));
$('event-select').addEventListener('change',event=>{const id=Number(event.target.value);if(id===selectedEvent)return;selectedEvent=id;historyVersion++;startSelectionSwitch();render();drawHistory([]);$('history-detail').textContent='正在读取活动记录……';refresh();});
$('refresh').addEventListener('click',refresh);
$('event-banner').addEventListener('load',()=>{$('event-banner').hidden=false;});
$('event-banner').addEventListener('error',()=>{$('event-banner').hidden=true;$('event-banner').parentElement.dataset.placeholder='活动横幅暂不可用';});
for(const source of [$('event-select'),$('rank-select')])ForecastUI.enhanceSelect(source);
async function poll(){await refresh();setTimeout(poll,snapshot?.service?.mode==='replay'&&snapshot.service.state!=='complete'?1000:15000);}
poll();
setInterval(renderCountdown,1000);
window.addEventListener('resize',()=>drawHistory(lastPoints,lastCurve,lastTerminal,lastCheckpoints));
$('checkpoint-comparison').addEventListener('toggle',()=>drawHistory(lastPoints,lastCurve,lastTerminal,lastCheckpoints));

/* Reusable controls with native state sources and fully styled popups. */
'use strict';
window.ForecastUI = (() => {
  const controls = new Map();
  const views = new WeakMap();
  const svg = path => {
    const icon = document.createElementNS('http://www.w3.org/2000/svg','svg');
    icon.setAttribute('viewBox','0 0 20 20'); icon.setAttribute('aria-hidden','true');
    icon.setAttribute('fill','none'); icon.setAttribute('stroke','currentColor');
    icon.setAttribute('stroke-width','1.8'); icon.setAttribute('stroke-linecap','round');
    icon.setAttribute('stroke-linejoin','round');
    const shape=document.createElementNS(icon.namespaceURI,'path');shape.setAttribute('d',path);icon.append(shape);
    return icon;
  };
  let opened=null;
  class SelectControl {
    constructor(source) {
      this.source=source; this.active=0; this.search='';this.searchedAt=0; this.value=null;
      this.root=document.createElement('div');this.root.className='ui-select';
      this.trigger=document.createElement('button');this.trigger.type='button';this.trigger.className='ui-select-trigger';
      this.trigger.id=source.id+'-trigger';this.trigger.setAttribute('role','combobox');
      this.trigger.setAttribute('aria-haspopup','listbox');this.trigger.setAttribute('aria-expanded','false');
      const label=document.querySelector(`label[for="${source.id}"]`);
      this.trigger.setAttribute('aria-label',source.getAttribute('aria-label')||label?.textContent.trim()||'选择');
      if(label)label.htmlFor=this.trigger.id;
      this.caption=document.createElement('span');this.caption.className='ui-select-value';
      this.trigger.append(this.caption,svg('m5 8 5 5 5-5'));
      this.menu=document.createElement('ul');this.menu.className='ui-select-menu';this.menu.hidden=true;
      this.menu.id=source.id+'-listbox';this.menu.setAttribute('role','listbox');
      this.menu.setAttribute('aria-label',this.trigger.getAttribute('aria-label'));
      this.trigger.setAttribute('aria-controls',this.menu.id);
      this.root.append(this.trigger,this.menu);source.after(this.root);source.hidden=true;
      this.trigger.addEventListener('click',()=>this.menu.hidden?this.open():this.close());
      this.trigger.addEventListener('keydown',event=>this.keydown(event));
      this.source.addEventListener('change',()=>this.sync());
      this.observer=new MutationObserver(()=>this.sync());
      this.observer.observe(source,{childList:true,subtree:true,attributes:true});
      this.layoutObserver=new ResizeObserver(()=>{if(opened===this)this.placeMenu();});
      this.layoutObserver.observe(this.root.closest('.overview')||this.root.parentElement);
      this.sync();
    }
    items(){return [...this.source.options].filter(option=>!option.disabled);}
    sync(){
      const items=this.items(), signature=items.map(option=>[option.value,option.textContent]);
      const key=JSON.stringify(signature);
      if(this.signature!==key){
        const activeValue=this.menu.children[this.active]?.dataset.value;
        this.signature=key;this.menu.replaceChildren();
        items.forEach((option,index)=>{
          const item=document.createElement('li');item.id=this.source.id+'-option-'+index;
          item.setAttribute('role','option');item.dataset.value=option.value;item.className='ui-select-option';
          const text=document.createElement('span');text.textContent=option.textContent;
          item.append(text,svg('m4 10 4 4 8-8'));
          item.addEventListener('pointerdown',event=>event.preventDefault());
          item.addEventListener('click',()=>this.choose(index));
          item.addEventListener('pointermove',()=>this.highlight(index,false));
          this.menu.append(item);
        });
        this.active=Math.max(0,items.findIndex(option=>option.value===activeValue));
      }
      const current=this.source.selectedOptions[0];
      this.caption.textContent=current?.textContent||'暂无可选项';
      this.trigger.setAttribute('aria-valuetext',current?.textContent||'');
      this.trigger.disabled=this.source.disabled||!items.length;
      [...this.menu.children].forEach(item=>item.setAttribute('aria-selected',String(item.dataset.value===this.source.value)));
      if(this.trigger.disabled)this.close();
      if(!this.menu.hidden)this.highlight(this.active,false);
    }
    open(){
      if(this.trigger.disabled)return;
      if(opened&&opened!==this)opened.close();
      opened=this;this.menu.hidden=false;this.trigger.setAttribute('aria-expanded','true');
      this.root.closest('.panel')?.classList.add('has-open-control');
      this.placeMenu();
      this.active=Math.max(0,this.items().findIndex(option=>option.value===this.source.value));
      this.highlight(this.active);
    }
    placeMenu(){
      const rect=this.trigger.getBoundingClientRect(),height=Math.min(this.menu.scrollHeight,280)+9;
      this.root.classList.toggle('opens-up',rect.bottom+height>innerHeight&&rect.top>height);
    }
    close(){
      this.menu.hidden=true;this.trigger.setAttribute('aria-expanded','false');
      this.root.closest('.panel')?.classList.remove('has-open-control');
      this.trigger.removeAttribute('aria-activedescendant');this.search='';
      if(opened===this)opened=null;
    }
    highlight(index,scroll=true){
      const items=[...this.menu.children];if(!items.length)return;
      this.active=(index+items.length)%items.length;
      items.forEach((item,i)=>item.classList.toggle('is-active',i===this.active));
      this.trigger.setAttribute('aria-activedescendant',items[this.active].id);
      if(scroll){
        const item=items[this.active].getBoundingClientRect(),menu=this.menu.getBoundingClientRect();
        if(item.top<menu.top+5)this.menu.scrollTop-=menu.top+5-item.top;
        else if(item.bottom>menu.bottom-5)this.menu.scrollTop+=item.bottom-menu.bottom+5;
      }
    }
    choose(index){
      const option=this.items()[index];if(!option)return;
      this.source.value=option.value;this.close();this.sync();
      this.source.dispatchEvent(new Event('change',{bubbles:true}));this.trigger.focus({preventScroll:true});
    }
    keydown(event){
      const visible=!this.menu.hidden;
      if(event.key==='Tab'){this.close();return;}
      if(event.key==='Escape'){if(visible){event.preventDefault();this.close();}return;}
      if(['ArrowDown','ArrowUp','Home','End'].includes(event.key)){
        event.preventDefault();if(!visible)this.open();
        if(event.key==='Home')this.highlight(0);
        else if(event.key==='End')this.highlight(this.items().length-1);
        else if(visible)this.highlight(this.active+(event.key==='ArrowDown'?1:-1));
        return;
      }
      if(event.key==='Enter'||event.key===' '){event.preventDefault();visible?this.choose(this.active):this.open();return;}
      if(event.key.length===1&&!event.ctrlKey&&!event.metaKey&&!event.altKey){
        event.preventDefault();if(!visible)this.open();
        const now=Date.now();this.search=(now-this.searchedAt<650?this.search:'')+event.key.toLowerCase();this.searchedAt=now;
        const items=this.items();
        const exact=items.findIndex(option=>{
          const label=option.textContent.toLowerCase();
          return label===this.search||(/^\d/.test(this.search)&&label.replace(/^t/,'')===this.search);
        });
        if(exact>=0){this.highlight(exact);return;}
        for(let offset=1;offset<=items.length;offset++){
          const index=(this.active+offset)%items.length;
          const label=items[index].textContent.toLowerCase();
          if(label.startsWith(this.search)||(/^\d/.test(this.search)&&label.replace(/^t/,'').startsWith(this.search))){this.highlight(index);break;}
        }
      }
    }
  }
  document.addEventListener('pointerdown',event=>{if(opened&&!opened.root.contains(event.target))opened.close();});
  window.addEventListener('resize',()=>opened?.close());
  return {
    enhanceSelect(source){if(!controls.has(source))controls.set(source,new SelectControl(source));return controls.get(source);},
    sync(){for(const control of controls.values())control.sync();},
    startView(root){
      const previous=views.get(root);
      if(previous?.animation?.playState==='running'){
        const current=getComputedStyle(root);root.style.opacity=current.opacity;root.style.transform=current.transform;
        previous.animation.cancel();
      }
      root.dataset.viewState='loading';root.setAttribute('aria-busy','true');
    },
    finishView(root){
      const previous=views.get(root);previous?.animation?.cancel();
      root.removeAttribute('aria-busy');
      if(matchMedia('(prefers-reduced-motion: reduce)').matches){
        root.style.removeProperty('opacity');root.style.removeProperty('transform');root.dataset.viewState='idle';views.delete(root);return;
      }
      const from={opacity:root.style.opacity||0,transform:root.style.transform||'translateY(6px)'};
      const animation=root.animate([from,{opacity:1,transform:'translateY(0)'}],{duration:220,easing:'cubic-bezier(.2,.65,.25,1)'});
      root.style.removeProperty('opacity');root.style.removeProperty('transform');
      root.dataset.viewState='entering';views.set(root,{animation});
      animation.finished.then(()=>{if(views.get(root)?.animation===animation){root.dataset.viewState='idle';views.delete(root);}}).catch(()=>{});
    }
  };
})();

/* Shared SVG reveal and cursor readout; values follow the rendered polylines. */
window.ForecastChart = (() => {
  let active=null, played=false, sequence=0;
  const element=(tag,attributes={})=>{
    const node=document.createElementNS('http://www.w3.org/2000/svg',tag);
    for(const [name,value] of Object.entries(attributes))node.setAttribute(name,String(value));
    return node;
  };
  function reading(series,time,extend=false){
    if(!series.length||time<series[0].time)return null;
    let lo=0,hi=series.length;
    while(lo<hi){const mid=(lo+hi)>>1;if(series[mid].time<time)lo=mid+1;else hi=mid;}
    if(lo<series.length&&series[lo].time===time)return {value:series[lo].value};
    if(lo===series.length)return extend?{value:series.at(-1).value,extended:true}:null;
    const a=series[lo-1],b=series[lo];
    return {value:a.value+(b.value-a.value)*(time-a.time)/(b.time-a.time),interpolated:true};
  }
  class Interaction {
    constructor(options){
      Object.assign(this,options);this.cursorTime=null;this.listeners=[];
      this.observed=options.observed.map(p=>({time:p.time,value:p.ep})).sort((a,b)=>a.time-b.time);
      this.predicted=options.predicted.map(p=>({time:p.observed_at,value:p.predicted_final_ep})).sort((a,b)=>a.time-b.time);
      this.times=[...new Set([this.first,...this.observed.map(p=>p.time),...this.predicted.map(p=>p.time),this.last])].sort((a,b)=>a-b);
      this.tooltip=document.getElementById('chart-tooltip');this.baseLabel=this.svg.getAttribute('aria-label');
      this.motion=matchMedia('(prefers-reduced-motion: reduce)');
      this.hover=element('g',{class:'chart-hover',display:'none','pointer-events':'none'});
      this.hover.append(element('line',{x1:0,x2:0,y1:this.top,y2:this.bottom,class:'chart-crosshair'}));
      this.observedDot=element('circle',{r:4.5,cx:0,cy:0,class:'chart-hover-point chart-observed-point'});
      this.predictedDot=element('circle',{r:4.5,cx:0,cy:0,class:'chart-hover-point chart-predicted-point'});
      this.hover.append(this.observedDot,this.predictedDot);this.svg.append(this.hover);
      this.on(this.svg,'pointermove',event=>{
        this.pendingPointer={clientX:event.clientX,clientY:event.clientY};
        if(!this.pointerFrame)this.pointerFrame=requestAnimationFrame(()=>{this.pointerFrame=0;if(!this.disposed)this.pointer(this.pendingPointer);});
      });
      this.on(this.svg,'pointerdown',event=>this.pointer(event));
      this.on(this.svg,'pointerleave',event=>{if(event.pointerType!=='touch')this.hide();});
      this.on(this.svg,'pointercancel',()=>this.hide());
      this.on(this.svg,'focus',()=>{this.finish();this.show(this.cursorTime??Math.max(this.observed.at(-1)?.time||this.first,this.predicted.at(-1)?.time||this.first));});
      this.on(this.svg,'blur',()=>this.hide());
      this.on(this.svg,'keydown',event=>this.keydown(event));
      this.on(document,'pointerdown',event=>{if(!this.svg.contains(event.target))this.hide();});
      this.on(document,'visibilitychange',()=>{if(document.hidden){this.finish();this.hide();}});
      this.on(this.motion,'change',()=>{if(this.motion.matches)this.finish();});
      this.reveal();
    }
    on(target,name,listener){target.addEventListener(name,listener);this.listeners.push(()=>target.removeEventListener(name,listener));}
    reveal(){
      if(played||this.motion.matches||document.hidden){played=true;this.svg.dataset.revealState='complete';return;}
      const nodes=[...this.svg.children].filter(node=>node.matches('.observed-area,.observed,.predicted:not(.projected),.chart-last-point'));
      if(!nodes.length){this.svg.dataset.revealState='complete';return;}
      this.series=element('g',{class:'chart-series'});this.svg.insertBefore(this.series,nodes[0]);
      this.series.append(...nodes);
      this.defs=element('defs');const clip=element('clipPath',{id:'chart-reveal-'+(++sequence),clipPathUnits:'userSpaceOnUse'});
      this.rect=element('rect',{x:this.left,y:this.top-5,width:this.right-this.left,height:this.bottom-this.top+10,class:'chart-reveal-rect',transform:`translate(${this.left} 0) scale(0 1) translate(${-this.left} 0)`});
      clip.append(this.rect);this.defs.append(clip);this.svg.prepend(this.defs);
      this.series.setAttribute('clip-path',`url(#${clip.id})`);
      this.projection=this.svg.querySelector('.projected');if(this.projection)this.projection.style.opacity='0';
      this.svg.dataset.revealState='pending';
      const start=()=>{
        if(this.disposed||this.svg.dataset.revealState!=='pending')return;
        played=true;this.observer?.disconnect();this.svg.dataset.revealState='playing';
        const endTime=Math.max(this.observed.at(-1)?.time||this.first,this.predicted.at(-1)?.time||this.first);
        const endpoint=Math.max(0,Math.min(1,(endTime-this.first)/(this.last-this.first)));
        const began=performance.now();
        const frame=now=>{
          if(this.disposed)return;
          const progress=Math.min(1,(now-began)/1400),scale=endpoint*(1-(1-progress)**2);
          this.rect.setAttribute('transform',`translate(${this.left} 0) scale(${scale} 1) translate(${-this.left} 0)`);
          if(progress<1)this.frame=requestAnimationFrame(frame);else this.finish(true);
        };
        this.frame=requestAnimationFrame(frame);
      };
      if('IntersectionObserver' in window){this.observer=new IntersectionObserver(entries=>{if(entries.some(entry=>entry.isIntersecting))start();},{threshold:.06});this.observer.observe(this.svg);}else start();
    }
    finish(fade=false){
      played=true;cancelAnimationFrame(this.frame);this.observer?.disconnect();
      this.series?.removeAttribute('clip-path');this.defs?.remove();
      if(this.projection){
        this.fade?.cancel();this.projection.style.removeProperty('opacity');
        if(fade&&!this.motion.matches)this.fade=this.projection.animate([{opacity:0},{opacity:.65}],{duration:180,easing:'ease-out'});
      }
      this.svg.dataset.revealState='complete';
    }
    pointer(event){
      const rect=this.svg.getBoundingClientRect(),view=this.svg.viewBox.baseVal;
      const px=(event.clientX-rect.left)*view.width/rect.width,py=(event.clientY-rect.top)*view.height/rect.height;
      if(px<this.left||px>this.right||py<this.top||py>this.bottom){this.hide();return;}
      this.finish();this.show(this.first+(px-this.left)/(this.right-this.left)*(this.last-this.first),py);
    }
    show(time,pointerY=this.top+24){
      const t=Math.round(Math.max(this.first,Math.min(this.last,time)));this.cursorTime=t;
      const actual=reading(this.observed,t),prediction=reading(this.predicted,t,true);
      const xx=this.x(t);this.hover.removeAttribute('display');this.hover.setAttribute('transform',`translate(${xx} 0)`);
      for(const [dot,value] of [[this.observedDot,actual],[this.predictedDot,prediction]]){
        if(value){dot.removeAttribute('display');dot.setAttribute('cy',this.y(value.value));}else dot.setAttribute('display','none');
      }
      const currentText=actual?this.formatValue(actual.value):'暂无观测';
      const predictedText=prediction?this.formatValue(prediction.value):'暂无预测';
      this.tooltip.hidden=false;this.tooltip.dataset.timeMs=String(t);
      this.tooltip.querySelector('time').textContent=this.formatTime(t);
      this.tooltip.querySelector('[data-current]').textContent=currentText;
      this.tooltip.querySelector('[data-predicted]').textContent=predictedText;
      const note=document.getElementById('chart-tooltip-note');
      note.textContent=prediction?.extended?(this.projectionLabel||'当前预测延伸'):'';
      note.hidden=!prediction?.extended;
      const rect=this.svg.getBoundingClientRect(),view=this.svg.viewBox.baseVal;
      const tx=xx/view.width*rect.width,ty=pointerY/view.height*rect.height;
      const width=this.tooltip.offsetWidth,height=this.tooltip.offsetHeight;
      const left=Math.max(4,Math.min(tx+16+width<=rect.width?tx+16:tx-width-16,rect.width-width-4));
      const top=Math.max(4,Math.min(ty-18,rect.height-height-4));
      this.tooltip.style.transform=`translate3d(${left}px,${top}px,0)`;
      this.svg.setAttribute('aria-label',`${this.formatTime(t)}，当前分数线 ${currentText}，预测终分 ${predictedText}`);
    }
    keydown(event){
      if(event.key==='Escape'){event.preventDefault();this.hide();return;}
      if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;
      event.preventDefault();this.finish();
      const now=this.cursorTime??this.observed.at(-1)?.time??this.first;
      let target=now;
      if(event.key==='Home')target=this.first;
      else if(event.key==='End')target=this.last;
      else if(event.key==='ArrowRight')target=this.times.find(time=>time>now)??this.last;
      else target=[...this.times].reverse().find(time=>time<now)??this.first;
      this.show(target);
    }
    hide(){cancelAnimationFrame(this.pointerFrame);this.pointerFrame=0;this.hover.setAttribute('display','none');this.tooltip.hidden=true;this.svg.setAttribute('aria-label',this.baseLabel);}
    dispose(){
      this.disposed=true;cancelAnimationFrame(this.frame);cancelAnimationFrame(this.pointerFrame);this.observer?.disconnect();this.fade?.cancel();
      this.series?.removeAttribute('clip-path');this.defs?.remove();this.projection?.style.removeProperty('opacity');
      this.listeners.forEach(remove=>remove());this.hide();
    }
  }
  return {
    reset(svg){if(active){active.dispose();active=null;}svg.removeAttribute('data-reveal-state');},
    attach(options){if(active)active.dispose();active=new Interaction(options);}
  };
})();

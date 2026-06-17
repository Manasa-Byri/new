from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import logging
from functools import lru_cache
import os
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # project root
CSV_FILE_PATH = BASE_DIR / "USRW.NONX.IYM551ND.MEMBER.SWEEP.G2262V.csv"


class CSVInsuranceInsightsService:
    
    def __init__(self):
        self.csv_path = CSV_FILE_PATH
        self._df = None
        self._last_loaded = None
    
    def _load_data(self) -> pd.DataFrame:
        if self._df is None or self._should_reload():
            try:
                logger.info(f"Loading CSV data from {self.csv_path}")
                self._df = pd.read_csv(str(self.csv_path), low_memory=False)
                self._last_loaded = datetime.now()
                self._preprocess_data()
                logger.info(f"Loaded {len(self._df)} records from CSV")
            except Exception as e:
                logger.error(f"Error loading CSV: {str(e)}")
                raise
        return self._df
    
    def _should_reload(self) -> bool:
        if self._last_loaded is None:
            return True
        time_since_load = (datetime.now() - self._last_loaded).total_seconds()
        return time_since_load > 3600
    
    def _preprocess_data(self):
        df = self._df
        
        df['MEMBIRDT_DATE'] = pd.to_datetime(df['MEMBIRDT'].astype(str), format='%Y%m%d', errors='coerce')
        df['CONT_EFF_DATE'] = pd.to_datetime(df['CONT EFF'].astype(str), format='%Y%m%d', errors='coerce')
        df['MEMEFFDT_DATE'] = pd.to_datetime(df['MEMEFFDT'].astype(str), format='%Y%m%d', errors='coerce')
        
        df['CONT_CAN_DATE'] = df['CONT CAN'].apply(
            lambda x: pd.to_datetime(str(x), format='%Y%m%d', errors='coerce') if x != 0 else None
        )
        df['MEMCANDT_DATE'] = df['MEMCANDT'].apply(
            lambda x: pd.to_datetime(str(x), format='%Y%m%d', errors='coerce') if x != 0 else None
        )
        
        df['AGE'] = ((datetime.now() - df['MEMBIRDT_DATE']).dt.days / 365.25).fillna(0).astype(int)
        
        df['STATUS_LABEL'] = df['STS'].map({0: 'Active', 1: 'Inactive', 2: 'Other', 4: 'Other'})
        
        df['MEMBER_TYPE_LABEL'] = df['MCDE'].apply(lambda x: 
            'Primary Male' if x == 10 else 
            'Primary Female' if x == 20 else 
            'Dependent'
        )
    
    async def get_membership_summary(self) -> Dict[str, Any]:
        try:
            df = self._load_data()
            
            summary = {
                "total_members": int(len(df)),
                "active_members": int(len(df[df['STS'] == 0])),
                "inactive_members": int(len(df[df['STS'] == 1])),
                "total_certificates": int(df['CERT'].nunique()),
                "avg_family_size": round(float(df['MCNT'].mean()), 2),
                "total_states": int(df['ST'].nunique()),
                "medical_coverage": int(len(df[df['TYP'] == 'MED'])),
                "dental_coverage": int(len(df[df['TYP'] == 'DEN'])),
                "vision_coverage": int(len(df[df['TYP'] == 'VIS']))
            }
            
            if summary['total_members'] > 0:
                summary['active_rate'] = round((summary['active_members'] / summary['total_members']) * 100, 2)
            else:
                summary['active_rate'] = 0
            
            return {
                "success": True,
                "data": summary,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in get_membership_summary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    async def get_membership_by_state(self) -> Dict[str, Any]:
        try:
            df = self._load_data()
            
            state_stats = df.groupby('ST').agg({
                'CERT': 'count',
                'STS': lambda x: (x == 0).sum()
            }).reset_index()
            
            state_stats.columns = ['state', 'total_members', 'active_members']
            state_stats['inactive_members'] = state_stats['total_members'] - state_stats['active_members']
            state_stats['active_rate'] = round((state_stats['active_members'] / state_stats['total_members']) * 100, 2)
            
            state_stats = state_stats.sort_values('total_members', ascending=False)
            
            results = state_stats.to_dict('records')
            
            return {
                "success": True,
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error in get_membership_by_state: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_age_demographics(self) -> Dict[str, Any]:
        try:
            df = self._load_data()
            
            age_bins = [0, 18, 26, 35, 45, 55, 65, 100]
            age_labels = ['0-17', '18-25', '26-34', '35-44', '45-54', '55-64', '65+']
            
            df['AGE_GROUP'] = pd.cut(df['AGE'], bins=age_bins, labels=age_labels, right=False)
            
            age_dist = df['AGE_GROUP'].value_counts().sort_index()
            
            results = []
            total = len(df)
            for age_group, count in age_dist.items():
                results.append({
                    "age_group": str(age_group),
                    "count": int(count),
                    "percentage": round((count / total) * 100, 2)
                })
            
            avg_age = round(float(df['AGE'].mean()), 1)
            median_age = int(df['AGE'].median())
            
            return {
                "success": True,
                "data": {
                    "distribution": results,
                    "average_age": avg_age,
                    "median_age": median_age
                }
            }
        except Exception as e:
            logger.error(f"Error in get_age_demographics: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    async def get_member_type_distribution(self) -> Dict[str, Any]:
        try:
            df = self._load_data()
            
            type_dist = df['MEMBER_TYPE_LABEL'].value_counts()
            
            results = []
            total = len(df)
            for member_type, count in type_dist.items():
                results.append({
                    "member_type": member_type,
                    "count": int(count),
                    "percentage": round((count / total) * 100, 2)
                })
            
            return {
                "success": True,
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error in get_member_type_distribution: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_family_size_distribution(self) -> Dict[str, Any]:
        try:
            df = self._load_data()
            
            family_dist = df['MCNT'].value_counts().sort_index()
            
            results = []
            total = len(df)
            for size, count in family_dist.items():
                results.append({
                    "family_size": int(size),
                    "count": int(count),
                    "percentage": round((count / total) * 100, 2)
                })
            
            return {
                "success": True,
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error in get_family_size_distribution: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_plan_distribution(self) -> Dict[str, Any]:
        try:
            df = self._load_data()
            
            type_dist = df['TYP'].value_counts()
            subtype_dist = df['TP1'].value_counts()
            
            type_results = []
            total = len(df)
            for plan_type, count in type_dist.items():
                type_results.append({
                    "plan_type": plan_type,
                    "count": int(count),
                    "percentage": round((count / total) * 100, 2)
                })
            
            subtype_results = []
            for subtype, count in subtype_dist.items():
                if pd.notna(subtype):
                    subtype_results.append({
                        "plan_subtype": subtype,
                        "count": int(count),
                        "percentage": round((count / total) * 100, 2)
                    })
            
            return {
                "success": True,
                "data": {
                    "by_type": type_results,
                    "by_subtype": subtype_results
                }
            }
        except Exception as e:
            logger.error(f"Error in get_plan_distribution: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": None
            }
    
    async def get_plan_by_state(self) -> Dict[str, Any]:
        try:
            df = self._load_data()
            
            state_plan = df.groupby(['ST', 'TYP']).size().reset_index(name='count')
            state_plan = state_plan.sort_values(['ST', 'count'], ascending=[True, False])
            
            results = state_plan.to_dict('records')
            
            return {
                "success": True,
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error in get_plan_by_state: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_hmo_vs_ppo_analysis(self) -> Dict[str, Any]:
        try:
            df = self._load_data()
            
            hmo_ppo = df[df['TP1'].isin(['HMO', 'PPO'])]
            
            comparison = hmo_ppo.groupby('TP1').agg({
                'CERT': 'count',
                'STS': lambda x: (x == 0).sum()
            }).reset_index()
            
            comparison.columns = ['plan_type', 'total_members', 'active_members']
            comparison['inactive_members'] = comparison['total_members'] - comparison['active_members']
            comparison['active_rate'] = round((comparison['active_members'] / comparison['total_members']) * 100, 2)
            
            results = comparison.to_dict('records')
            
            return {
                "success": True,
                "data": results
            }
        except Exception as e:
            logger.error(f"Error in get_hmo_vs_ppo_analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_cancellation_reasons(self, limit: int = 20) -> Dict[str, Any]:
        try:
            df = self._load_data()
            
            cancelled = df[df['STS'] == 1]
            
            reason_map = {
                '11': 'Never Effective',
                '47': 'Unknown Reason',
                '08': 'Non-Payment',
                '06': 'Voluntary Termination',
                '07': 'Other Termination',
                '02': 'Coverage Change',
                '01': 'Death',
                '92': 'Administrative',
                '05': 'Moved Out of Area',
                '67': 'Other'
            }
            
            reason_counts = cancelled['CR'].value_counts().head(limit)
            
            results = []
            total = len(cancelled)
            for reason_code, count in reason_counts.items():
                if pd.notna(reason_code):
                    results.append({
                        "reason_code": str(reason_code),
                        "reason_description": reason_map.get(str(reason_code), "Unknown"),
                        "count": int(count),
                        "percentage": round((count / total) * 100, 2)
                    })
            
            return {
                "success": True,
                "data": results,
                "count": len(results),
                "total_cancellations": total
            }
        except Exception as e:
            logger.error(f"Error in get_cancellation_reasons: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_cancellation_by_plan_type(self) -> Dict[str, Any]:
        try:
            df = self._load_data()
            
            plan_cancel = df.groupby('TYP').agg({
                'CERT': 'count',
                'STS': lambda x: (x == 1).sum()
            }).reset_index()
            
            plan_cancel.columns = ['plan_type', 'total_members', 'cancelled_members']
            plan_cancel['active_members'] = plan_cancel['total_members'] - plan_cancel['cancelled_members']
            plan_cancel['cancellation_rate'] = round((plan_cancel['cancelled_members'] / plan_cancel['total_members']) * 100, 2)
            plan_cancel['retention_rate'] = round(100 - plan_cancel['cancellation_rate'], 2)
            
            plan_cancel = plan_cancel.sort_values('total_members', ascending=False)
            
            results = plan_cancel.to_dict('records')
            
            return {
                "success": True,
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error in get_cancellation_by_plan_type: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_business_segment_analysis(self) -> Dict[str, Any]:
        try:
            df = self._load_data()
            
            segment_stats = df.groupby('MBUTY').agg({
                'CERT': 'count',
                'STS': lambda x: (x == 0).sum()
            }).reset_index()
            
            segment_stats.columns = ['business_type', 'total_members', 'active_members']
            segment_stats['inactive_members'] = segment_stats['total_members'] - segment_stats['active_members']
            segment_stats['active_rate'] = round((segment_stats['active_members'] / segment_stats['total_members']) * 100, 2)
            segment_stats['retention_rate'] = segment_stats['active_rate']
            
            segment_stats = segment_stats.sort_values('total_members', ascending=False)
            
            results = segment_stats.to_dict('records')
            
            return {
                "success": True,
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error in get_business_segment_analysis: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_exchange_distribution(self) -> Dict[str, Any]:
        try:
            df = self._load_data()
            
            exchange_map = {
                'PB': 'Public Exchange',
                'PR': 'Private Exchange',
                'PO': 'Public Exchange',
                'OF': 'Off Exchange',
                'EL': 'Eligible'
            }
            
            df['EXCHANGE_LABEL'] = df['XI'].fillna('Non-Exchange').map(lambda x: exchange_map.get(x, x) if x != 'Non-Exchange' else 'Non-Exchange')
            
            exchange_dist = df['EXCHANGE_LABEL'].value_counts()
            
            results = []
            total = len(df)
            for exchange_type, count in exchange_dist.items():
                results.append({
                    "exchange_type": exchange_type,
                    "count": int(count),
                    "percentage": round((count / total) * 100, 2)
                })
            
            return {
                "success": True,
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error in get_exchange_distribution: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_provider_performance(self, limit: int = 20) -> Dict[str, Any]:
        try:
            df = self._load_data()
            
            provider_stats = df.groupby('PRVDR').agg({
                'CERT': 'count',
                'STS': lambda x: (x == 0).sum()
            }).reset_index()
            
            provider_stats.columns = ['provider_code', 'total_members', 'active_members']
            provider_stats['retention_rate'] = round((provider_stats['active_members'] / provider_stats['total_members']) * 100, 2)
            
            provider_stats = provider_stats.sort_values('total_members', ascending=False).head(limit)
            
            results = provider_stats.to_dict('records')
            
            return {
                "success": True,
                "data": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Error in get_provider_performance: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "data": []
            }
    
    async def get_consolidated_summary(self) -> Dict[str, Any]:
        try:
            df = self._load_data()
            
            total_members = len(df)
            active_members = len(df[df['STS'] == 0])
            active_rate = round((active_members / total_members) * 100, 1)
            
            medical_count = len(df[df['TYP'] == 'MED'])
            dental_count = len(df[df['TYP'] == 'DEN'])
            medical_pct = round((medical_count / total_members) * 100, 1)
            
            ppo_count = len(df[df['TP1'] == 'PPO'])
            hmo_count = len(df[df['TP1'] == 'HMO'])
            ppo_pct = round((ppo_count / total_members) * 100, 1)
            
            smgrp_count = len(df[df['MBUTY'] == 'SMGRP'])
            ind_count = len(df[df['MBUTY'] == 'IND'])
            smgrp_pct = round((smgrp_count / total_members) * 100, 1)
            
            top_states = df['ST'].value_counts().head(3)
            top_state_list = [f"{state} ({count:,})" for state, count in top_states.items()]
            
            cancelled = df[df['STS'] == 1]
            if len(cancelled) > 0:
                top_cancel_reason = cancelled['CR'].value_counts().head(1)
                cancel_code = str(top_cancel_reason.index[0]) if len(top_cancel_reason) > 0 else "N/A"
                cancel_count = int(top_cancel_reason.values[0]) if len(top_cancel_reason) > 0 else 0
                cancel_reason_map = {
                    '11': 'Never Effective',
                    '08': 'Non-Payment',
                    '47': 'Unknown',
                    '06': 'Voluntary Termination'
                }
                cancel_reason = cancel_reason_map.get(cancel_code, 'Other')
            else:
                cancel_reason = "N/A"
                cancel_count = 0
            
            avg_age = round(df['AGE'].mean(), 1)
            total_states = df['ST'].nunique()
            
            summary_text = (
                f"The insurance portfolio comprises {total_members:,} members across {total_states} states, "
                f"with an active membership rate of {active_rate}% ({active_members:,} active members). "
                f"Medical coverage dominates at {medical_pct}% ({medical_count:,} members), while dental coverage accounts for {dental_count:,} members. "
                f"Plan preferences show {ppo_pct}% favor PPO plans ({ppo_count:,} members) compared to HMO plans ({hmo_count:,} members), "
                f"with the member base split between Small Group ({smgrp_pct}%, {smgrp_count:,} members) and Individual ({ind_count:,} members) segments. "
                f"Geographic concentration is highest in {', '.join(top_state_list[:3])}, and the average member age is {avg_age} years. "
                f"Among inactive members, the primary cancellation reason is '{cancel_reason}' affecting {cancel_count:,} members, "
                f"indicating key areas for retention strategy focus."
            )
            
            return {
                "success": True,
                "response": summary_text,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in get_consolidated_summary: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "Unable to generate summary due to an error."
            }
    
    async def reload_data(self) -> Dict[str, Any]:
        try:
            self._df = None
            self._last_loaded = None
            df = self._load_data()
            
            return {
                "success": True,
                "message": "Data reloaded successfully",
                "records_loaded": len(df),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error reloading data: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
